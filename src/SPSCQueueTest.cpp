// SPDX-FileCopyrightText: 2018 Erik Rigtorp <erik@rigtorp.se>
// SPDX-License-Identifier: MIT

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <rigtorp/SPSCQueue.h>
#include <set>
#include <stdexcept>
#include <thread>
#include <type_traits>

#if (defined(__unix__) || defined(__APPLE__)) && !defined(NDEBUG)
#include <csignal>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(expr)                                                            \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK failed: " #expr     \
                << std::endl;                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (false)

// TestType tracks correct usage of constructors and destructors
struct TestType {
  static std::set<const TestType *> constructed;
  TestType() noexcept {
    CHECK(constructed.count(this) == 0);
    constructed.insert(this);
  };
  TestType(const TestType &other) noexcept {
    CHECK(constructed.count(this) == 0);
    CHECK(constructed.count(&other) == 1);
    constructed.insert(this);
  };
  TestType(TestType &&other) noexcept {
    CHECK(constructed.count(this) == 0);
    CHECK(constructed.count(&other) == 1);
    constructed.insert(this);
  };
  TestType &operator=(const TestType &other) noexcept {
    CHECK(constructed.count(this) == 1);
    CHECK(constructed.count(&other) == 1);
    return *this;
  };
  TestType &operator=(TestType &&other) noexcept {
    CHECK(constructed.count(this) == 1);
    CHECK(constructed.count(&other) == 1);
    return *this;
  }
  ~TestType() noexcept {
    CHECK(constructed.count(this) == 1);
    constructed.erase(this);
  };
};

std::set<const TestType *> TestType::constructed;

template <typename T> struct TrackingAllocator {
  using value_type = T;

  struct State {
    size_t allocations = 0;
    size_t deallocations = 0;
    T *lastPtr = nullptr;
    size_t lastCount = 0;
  };

  State *state;

  explicit TrackingAllocator(State *s) noexcept : state(s) {}

  template <typename U>
  TrackingAllocator(const TrackingAllocator<U> &other) noexcept
      : state(other.state) {}

  T *allocate(size_t n) {
    ++state->allocations;
    state->lastCount = n;
    state->lastPtr = std::allocator<T>{}.allocate(n);
    return state->lastPtr;
  }

  void deallocate(T *p, size_t n) noexcept {
    CHECK(p == state->lastPtr);
    CHECK(n == state->lastCount);
    ++state->deallocations;
    std::allocator<T>{}.deallocate(p, n);
  }
};

template <typename T> struct OversizeAllocator {
  using value_type = T;

  struct AllocationResult {
    T *ptr;
    size_t count;
  };

  struct State {
    size_t extra = 0;
    size_t requested = 0;
    size_t allocated = 0;
    size_t deallocated = 0;
    size_t reportedCount = 0;
  };

  State *state;

  explicit OversizeAllocator(State *s) noexcept : state(s) {}

  AllocationResult allocate_at_least(size_t n) {
    state->requested = n;
    state->allocated = n + state->extra;
    return {static_cast<T *>(::operator new(state->allocated * sizeof(T))),
            state->reportedCount ? state->reportedCount : state->allocated};
  }

  void deallocate(T *p, size_t n) noexcept {
    state->deallocated = n;
    ::operator delete(p);
  }
};

template <bool Closable> struct ClosureConfig : rigtorp::DefaultConfig {
  static constexpr bool completable = Closable;
};

constexpr size_t kCompletionBit = size_t{1}
                                  << (std::numeric_limits<size_t>::digits - 1);

template <rigtorp::WaitPolicy Producer, rigtorp::WaitPolicy Consumer,
          bool Closable>
struct WaitingConfig : ClosureConfig<Closable> {
  static constexpr auto producerWaitPolicy = Producer;
  static constexpr auto consumerWaitPolicy = Consumer;
};

static void testFunctionality() {
  using namespace rigtorp;

  {
    SPSCQueue<TestType> q(10);
    CHECK(q.front() == nullptr);
    CHECK(q.size() == 0);
    CHECK(q.empty() == true);
    CHECK(q.capacity() == 10);
    for (int i = 0; i < 10; i++) {
      q.emplace();
    }
    CHECK(q.front() != nullptr);
    CHECK(q.size() == 10);
    CHECK(q.empty() == false);
    CHECK(TestType::constructed.size() == 10);
    CHECK(q.try_emplace() == false);
    q.pop();
    CHECK(q.size() == 9);
    CHECK(TestType::constructed.size() == 9);
    q.pop();
    CHECK(q.size() == 8);
    CHECK(q.try_emplace() == true);
    // writeIdx_ has now wrapped past readIdx_
    CHECK(q.size() == 9);
    CHECK(q.empty() == false);
    CHECK(TestType::constructed.size() == 9);
    CHECK(q.try_emplace() == true);
    CHECK(q.size() == 10);
    CHECK(q.try_emplace() == false);
    CHECK(TestType::constructed.size() == 10);
    q.pop();
    CHECK(q.size() == 9);
    // Destroy non-empty wrapped-around queue
  }
  CHECK(TestType::constructed.size() == 0);

  // Zero and one both give minimal capacity across multiple wraparounds
  for (size_t capacity : {size_t{0}, size_t{1}}) {
    SPSCQueue<int> q(capacity);
    CHECK(q.capacity() == 1);
    for (int i = 0; i < 5; ++i) {
      CHECK(q.empty());
      CHECK(q.size() == 0);
      CHECK(q.front() == nullptr);
      CHECK(q.try_push(i));
      CHECK(!q.empty());
      CHECK(q.size() == 1);
      CHECK(!q.try_push(i + 1));
      CHECK(q.front() != nullptr);
      CHECK(*q.front() == i);
      q.pop();
    }
    CHECK(q.empty());
    CHECK(q.size() == 0);
  }
}

// Check overload participation without instantiating operation bodies.
namespace constraints_test {
using rigtorp::WaitPolicy;

template <typename Config, typename Allocator = std::allocator<int>>
concept QueueConfig =
    requires { typename rigtorp::SPSCQueue<int, Config, Allocator>; };

template <typename Queue>
concept CanClose = requires(Queue &q) {
  { q.complete() } noexcept;
};

template <typename Queue>
concept CanCheckClosed = requires(const Queue &q) {
  { q.completed() } noexcept -> std::same_as<bool>;
};

struct WrongPolicyType : rigtorp::DefaultConfig {
  static constexpr int producerWaitPolicy = 0;
};
struct WrongConsumerPolicyType : rigtorp::DefaultConfig {
  static constexpr int consumerWaitPolicy = 0;
};
struct WrongCloseType : rigtorp::DefaultConfig {
  static constexpr int completable = false;
};
static_assert(!QueueConfig<std::allocator<int>>);
static_assert(!QueueConfig<WrongPolicyType>);
static_assert(!QueueConfig<WrongConsumerPolicyType>);
static_assert(!QueueConfig<WrongCloseType>);
using ClosableQueue = rigtorp::SPSCQueue<
    int, WaitingConfig<WaitPolicy::Atomic, WaitPolicy::Atomic, true>>;
static_assert(CanClose<ClosableQueue>);
static_assert(CanCheckClosed<ClosableQueue>);
static_assert(!CanClose<rigtorp::SPSCQueue<int>>);
static_assert(!CanCheckClosed<rigtorp::SPSCQueue<int>>);
static_assert(requires(rigtorp::SPSCQueue<int> &q, int &value,
                       void (*fn)(int &) noexcept) {
  { q.wait_pop(value) } noexcept -> std::same_as<void>;
  { q.consume(fn) } noexcept -> std::same_as<void>;
});
static_assert(requires(ClosableQueue &q, int &value,
                       void (*fn)(int &) noexcept) {
  { q.wait_pop(value) } noexcept -> std::same_as<bool>;
  { q.consume(fn) } noexcept -> std::same_as<bool>;
});

template <typename T>
concept QueueElement = requires { typename rigtorp::SPSCQueue<T>; };
template <typename T, typename... Args>
concept CanEmplace = requires(rigtorp::SPSCQueue<T> &q, Args &&...args) {
  q.emplace(std::forward<Args>(args)...);
};
template <typename T, typename... Args>
concept CanTryEmplace = requires(rigtorp::SPSCQueue<T> &q, Args &&...args) {
  q.try_emplace(std::forward<Args>(args)...);
};
template <typename T, typename P>
concept CanPush = requires(rigtorp::SPSCQueue<T> &q, P &&value) {
  q.push(std::forward<P>(value));
};
template <typename T, typename P>
concept CanTryPush = requires(rigtorp::SPSCQueue<T> &q, P &&value) {
  q.try_push(std::forward<P>(value));
};
template <typename T>
concept CanTryPop =
    requires(rigtorp::SPSCQueue<T> &q, T &value) { q.try_pop(value); };
template <typename T>
concept CanWaitPop =
    requires(rigtorp::SPSCQueue<T> &q, T &value) { q.wait_pop(value); };
template <typename T, typename Fn>
concept CanConsume = requires(rigtorp::SPSCQueue<T> &q, Fn &&fn) {
  q.try_consume(std::forward<Fn>(fn));
};
template <typename T, typename Fn>
concept CanBlockingConsume = requires(rigtorp::SPSCQueue<T> &q, Fn &&fn) {
  q.consume(std::forward<Fn>(fn));
};
struct Immovable {
  explicit Immovable(int) noexcept {}
  Immovable(const Immovable &) = delete;
  Immovable(Immovable &&) = delete;
  Immovable &operator=(const Immovable &) = delete;
  Immovable &operator=(Immovable &&) = delete;
  void consume() noexcept {}
};
struct CopyOnly {
  CopyOnly() = default;
  CopyOnly(const CopyOnly &) = default;
  CopyOnly(CopyOnly &&) = delete;
  CopyOnly &operator=(const CopyOnly &) = default;
  CopyOnly &operator=(CopyOnly &&) = delete;
};
struct ThrowingDestructor {
  ~ThrowingDestructor() noexcept(false) {}
};
struct RvalueCallback {
  void operator()(Immovable &) && noexcept {}
};
static_assert(!QueueElement<ThrowingDestructor>);
static_assert(QueueElement<Immovable>);
static_assert(CanEmplace<Immovable, int>);
static_assert(CanTryEmplace<Immovable, int>);
static_assert(!CanEmplace<Immovable>);
static_assert(!CanTryEmplace<Immovable>);
static_assert(!CanEmplace<Immovable, const char *>);
static_assert(!CanTryEmplace<Immovable, const char *>);
static_assert(!CanPush<Immovable, const Immovable &>);
static_assert(!CanTryPush<Immovable, const Immovable &>);
static_assert(CanPush<CopyOnly, CopyOnly>);
static_assert(CanTryPush<CopyOnly, CopyOnly>);
static_assert(CanTryPop<CopyOnly>);
static_assert(!CanTryPop<Immovable>);
static_assert(CanWaitPop<CopyOnly>);
static_assert(!CanWaitPop<Immovable>);
using MoveOnly = std::unique_ptr<int>;
static_assert(CanPush<MoveOnly, MoveOnly>);
static_assert(CanTryPush<MoveOnly, MoveOnly>);
static_assert(!CanPush<MoveOnly, const MoveOnly &>);
static_assert(!CanTryPush<MoveOnly, const MoveOnly &>);
static_assert(CanTryPop<MoveOnly>);
static_assert(CanWaitPop<MoveOnly>);
static_assert(CanConsume<Immovable, decltype(&Immovable::consume)>);
static_assert(CanConsume<Immovable, RvalueCallback>);
static_assert(!CanConsume<Immovable, RvalueCallback &>);
static_assert(!CanConsume<Immovable, int>);
static_assert(!CanConsume<Immovable, void (*)(int &)>);
static_assert(CanBlockingConsume<Immovable, decltype(&Immovable::consume)>);
static_assert(CanBlockingConsume<Immovable, RvalueCallback>);
static_assert(!CanBlockingConsume<Immovable, RvalueCallback &>);
static_assert(!CanBlockingConsume<Immovable, int>);
static_assert(!CanBlockingConsume<Immovable, void (*)(int &)>);
} // namespace constraints_test

static void testTypeCategories() {
  using namespace rigtorp;

  // Copyable only type
  {
    struct Test {
      Test() {}
      Test(const Test &) {}
      Test(Test &&) = delete;
    };
    SPSCQueue<Test> q(16);
    // lvalue
    Test v;
    q.emplace(v);
    (void)q.try_emplace(v);
    q.push(v);
    (void)q.try_push(v);
    static_assert(noexcept(q.emplace(v)) == false, "");
    static_assert(noexcept(q.try_emplace(v)) == false, "");
    static_assert(noexcept(q.push(v)) == false, "");
    static_assert(noexcept(q.try_push(v)) == false, "");
    // xvalue
    q.push(Test());
    (void)q.try_push(Test());
    static_assert(noexcept(q.push(Test())) == false, "");
    static_assert(noexcept(q.try_push(Test())) == false, "");
  }

  // Copyable only type (noexcept)
  {
    struct Test {
      Test() noexcept {}
      Test(const Test &) noexcept {}
      Test(Test &&) = delete;
    };
    SPSCQueue<Test> q(16);
    // lvalue
    Test v;
    q.emplace(v);
    (void)q.try_emplace(v);
    q.push(v);
    (void)q.try_push(v);
    static_assert(noexcept(q.emplace(v)) == true, "");
    static_assert(noexcept(q.try_emplace(v)) == true, "");
    static_assert(noexcept(q.push(v)) == true, "");
    static_assert(noexcept(q.try_push(v)) == true, "");
    // xvalue
    q.push(Test());
    (void)q.try_push(Test());
    static_assert(noexcept(q.push(Test())) == true, "");
    static_assert(noexcept(q.try_push(Test())) == true, "");
  }

  // Movable only type
  {
    SPSCQueue<std::unique_ptr<int>> q(16);
    // xvalue
    q.emplace(std::unique_ptr<int>(new int(1)));
    (void)q.try_emplace(std::unique_ptr<int>(new int(1)));
    q.push(std::unique_ptr<int>(new int(1)));
    (void)q.try_push(std::unique_ptr<int>(new int(1)));
    auto v = std::unique_ptr<int>(new int(1));
    static_assert(noexcept(q.emplace(std::move(v))) == true, "");
    static_assert(noexcept(q.try_emplace(std::move(v))) == true, "");
    static_assert(noexcept(q.push(std::move(v))) == true, "");
    static_assert(noexcept(q.try_push(std::move(v))) == true, "");
  }

  // Non-default-constructible, non-copyable, non-movable type with multi-arg
  // emplace
  {
    struct Immovable {
      int a;
      int b;
      Immovable(int x, int y) noexcept : a(x), b(y) {}
      Immovable(const Immovable &) = delete;
      Immovable(Immovable &&) = delete;
    };
    SPSCQueue<Immovable> q(4);
    q.emplace(1, 2);
    CHECK(q.try_emplace(3, 4));
    CHECK(q.front()->a == 1 && q.front()->b == 2);
    q.pop();
    CHECK(q.front()->a == 3 && q.front()->b == 4);
    q.pop();
    CHECK(q.empty());
  }

  // Small (sizeof == 1) and large (sizeof > kFalseSharingRange) element types
  {
    SPSCQueue<uint8_t> smallQ(3);
    CHECK(smallQ.try_push(uint8_t{42}));
    CHECK(*smallQ.front() == 42);
    smallQ.pop();

    struct Large {
      char buf[512];
      explicit Large(char c) noexcept {
        buf[0] = c;
        buf[511] = c;
      }
    };
    SPSCQueue<Large> largeQ(2);
    largeQ.emplace('z');
    CHECK(largeQ.front()->buf[0] == 'z');
    CHECK(largeQ.front()->buf[511] == 'z');
    largeQ.pop();
  }
}

static void testExceptionSafety() {
  using namespace rigtorp;

  struct TrackedThrow {
    int *liveCount;
    int value;
    TrackedThrow(int *live, int v, bool shouldThrow)
        : liveCount(live), value(v) {
      if (shouldThrow) {
        throw std::runtime_error("boom");
      }
      ++(*liveCount);
    }
    ~TrackedThrow() noexcept { --(*liveCount); }
  };

  int live = 0;
  {
    SPSCQueue<TrackedThrow> q(3);
    q.emplace(&live, 10, false);
    CHECK(live == 1);
    CHECK(q.size() == 1);

    bool threw = false;
    try {
      q.emplace(&live, 20, true);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    CHECK(live == 1);
    CHECK(q.size() == 1);

    threw = false;
    try {
      (void)q.try_emplace(&live, 30, true);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    CHECK(live == 1);
    CHECK(q.size() == 1);

    q.emplace(&live, 40, false);
    CHECK(live == 2);
    CHECK(q.size() == 2);
    CHECK(q.front()->value == 10);
    q.pop();
    CHECK(q.front()->value == 40);
    q.pop();
    CHECK(q.empty());
  }
  CHECK(live == 0);
}

static void testCustomAllocator() {
  using namespace rigtorp;

  TrackingAllocator<int>::State state;
  {
    SPSCQueue<int, DefaultConfig, TrackingAllocator<int>> q(
        5, TrackingAllocator<int>(&state));
    CHECK(state.allocations == 1);
    CHECK(state.deallocations == 0);
    CHECK(q.capacity() == 5);
    for (int i = 0; i < 5; ++i) {
      CHECK(q.try_push(i));
    }
    CHECK(!q.try_push(99));
    for (int i = 0; i < 5; ++i) {
      CHECK(*q.front() == i);
      q.pop();
    }
  }
  CHECK(state.deallocations == 1);

  OversizeAllocator<int>::State oversizeState;
  oversizeState.extra = 3;
  {
    SPSCQueue<int, DefaultConfig, OversizeAllocator<int>> q(
        4, OversizeAllocator<int>(&oversizeState));
    CHECK(q.capacity() == 7);
    for (int i = 0; i < 7; ++i) {
      CHECK(q.try_push(i));
    }
    CHECK(!q.try_push(100));
    for (int i = 0; i < 7; ++i) {
      CHECK(*q.front() == i);
      q.pop();
    }
  }
  CHECK(oversizeState.deallocated == oversizeState.allocated);

  // Exercise capacity capping without accessing the simulated extra slots.
  oversizeState.reportedCount = SIZE_MAX;
  {
    SPSCQueue<int, ClosureConfig<true>, OversizeAllocator<int>> q(
        4, OversizeAllocator<int>(&oversizeState));
    CHECK(q.capacity() == kCompletionBit - 1);
  }
  CHECK(oversizeState.deallocated == SIZE_MAX);
}

// Adapted from PR #51: inspect allocation arithmetic without large allocations.
template <typename T> struct RecordingAllocator {
  using value_type = T;
  size_t *requested;

  T *allocate(size_t n) {
    *requested = n;
    if (n != 0) {
      throw std::bad_alloc();
    }
    return nullptr;
  }

  void deallocate(T *, size_t) noexcept {}
};

template <typename T> struct RecordingAtLeastAllocator : RecordingAllocator<T> {
  struct AllocationResult {
    T *ptr;
    size_t count;
  };

  AllocationResult allocate_at_least(size_t n) {
    return {this->allocate(n), n};
  }
};

// By default test char, which needs one false-sharing range at each end.
template <typename Allocator, bool Closable = false>
static void testCapacityAllocationBounds(
    size_t padding = alignof(rigtorp::SPSCQueue<char>)) {
  using T = typename Allocator::value_type;
  const size_t maxCapacity =
      Closable ? kCompletionBit - 1 : SIZE_MAX - 2 * padding - 1;
  size_t requested = 0;
  Allocator allocator;
  allocator.requested = &requested;

  for (size_t capacity : {size_t{0}, size_t{1}, maxCapacity - 1, maxCapacity,
                          maxCapacity + 1, SIZE_MAX - 1, SIZE_MAX}) {
    requested = 0;
    bool threw = false;
    try {
      rigtorp::SPSCQueue<T, ClosureConfig<Closable>, Allocator> q(capacity,
                                                                  allocator);
    } catch (const std::bad_alloc &) {
      threw = true;
    }
    CHECK(threw);
    const size_t clamped =
        capacity == 0 ? 1 : (capacity > maxCapacity ? maxCapacity : capacity);
    const size_t expected = clamped + 1 + 2 * padding;
    CHECK(requested == expected);
  }
}

static void testAlignedElementPadding() {
  constexpr size_t range = alignof(rigtorp::SPSCQueue<char>);
  struct alignas(range) Aligned {
    explicit Aligned(int v = 0) : value(v) {}
    int value;
  };
  struct alignas(2 * range) OverAligned {
    explicit OverAligned(int v = 0) : value(v) {}
    int value;
  };
  struct Unaligned {
    explicit Unaligned(int v = 0) : value(v) {}
    int value;
    char bytes[range - sizeof(int)]{};
  };
  static_assert(sizeof(Unaligned) == range);
  static_assert(alignof(Unaligned) < range);

  auto check = []<typename T>(size_t padding) {
    testCapacityAllocationBounds<RecordingAllocator<T>>(padding);
    testCapacityAllocationBounds<RecordingAtLeastAllocator<T>>(padding);
    for (size_t capacity : {size_t{0}, size_t{1}, size_t{3}}) {
      typename TrackingAllocator<T>::State state;
      {
        rigtorp::SPSCQueue<T, rigtorp::DefaultConfig, TrackingAllocator<T>> q(
            capacity, TrackingAllocator<T>(&state));
        const size_t actual = capacity == 0 ? 1 : capacity;
        CHECK(q.capacity() == actual);
        CHECK(state.lastCount == actual + 1 + 2 * padding);
        for (int cycle = 0; cycle < 3; ++cycle) {
          for (size_t i = 0; i < actual; ++i) {
            CHECK(q.try_emplace(static_cast<int>(i)));
          }
          CHECK(!q.try_emplace());
          CHECK(q.front() != nullptr);
          CHECK(reinterpret_cast<uintptr_t>(q.front()) % alignof(T) == 0);
          for (size_t i = 0; i < actual; ++i) {
            CHECK(q.front() != nullptr);
            CHECK(q.front()->value == static_cast<int>(i));
            q.pop();
          }
          CHECK(q.front() == nullptr);
        }
        q.emplace(); // Exercise destruction of a non-empty queue.
      }
      CHECK(state.allocations == 1);
      CHECK(state.deallocations == 1);
    }
  };
  check.template operator()<Aligned>(0);
  check.template operator()<OverAligned>(0);
  // Size alone is insufficient: this type still needs padding at both ends.
  check.template operator()<Unaligned>(1);
}

static void testCapacityBounds() {
  using namespace rigtorp;

  // capacity < 1
  {
    SPSCQueue<int> q(0);
    CHECK(q.capacity() == 1);
  }

  // Check that padding doesn't overflow capacity
  {
    bool throws = false;
    try {
      SPSCQueue<int> q(SIZE_MAX - 1);
    } catch (...) {
      throws = true;
    }
    CHECK(throws);
  }

  {
    bool throws = false;
    try {
      SPSCQueue<int> q(SIZE_MAX);
    } catch (...) {
      throws = true;
    }
    CHECK(throws);
  }
}

static void testConcurrency() {
  using namespace rigtorp;

  // Fuzz and performance test
  {
    const size_t iter = 100000;
    SPSCQueue<size_t> q(iter / 1000 + 1);
    std::atomic<bool> flag(false);
    std::thread producer([&] {
      while (!flag)
        ;
      for (size_t i = 0; i < iter; ++i) {
        q.emplace(i);
      }
    });

    size_t sum = 0;
    auto start = std::chrono::system_clock::now();
    flag = true;
    for (size_t i = 0; i < iter; ++i) {
      while (!q.front())
        ;
      sum += *q.front();
      q.pop();
    }
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    CHECK(q.front() == nullptr);
    CHECK(sum == iter * (iter - 1) / 2);

    producer.join();

    std::cout << duration.count() / static_cast<int64_t>(iter) << " ns/iter"
              << std::endl;
  }

  // Contended try_emplace / try_push with multi-word payload and small ring
  {
    struct Payload {
      size_t seq;
      size_t check;
      explicit Payload(size_t s) noexcept : seq(s), check(s ^ 0x5a5a5a5a) {}
    };

    const size_t iter = 50000;
    SPSCQueue<Payload> q(7);
    std::thread producer([&] {
      for (size_t i = 0; i < iter; ++i) {
        while (!q.try_emplace(i)) {
          CHECK(q.size() <= q.capacity());
        }
      }
    });

    for (size_t i = 0; i < iter; ++i) {
      Payload *p = nullptr;
      while ((p = q.front()) == nullptr) {
        CHECK(q.size() <= q.capacity());
      }
      CHECK(p->seq == i);
      CHECK(p->check == (i ^ 0x5a5a5a5a));
      q.pop();
    }

    producer.join();
    CHECK(q.empty());
    CHECK(q.size() == 0);
  }

  // Blocking emplace unblocks when consumer drains a full queue
  {
    SPSCQueue<int> q(2);
    q.push(1);
    q.push(2);

    std::atomic<bool> pushedThird(false);
    std::thread consumer([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      CHECK(*q.front() == 1);
      q.pop();
      CHECK(*q.front() == 2);
      q.pop();
    });

    q.push(3);
    pushedThird.store(true);
    consumer.join();

    CHECK(pushedThird.load());
    CHECK(*q.front() == 3);
    q.pop();
    CHECK(q.empty());
  }
}

template <rigtorp::WaitPolicy Producer, rigtorp::WaitPolicy Consumer,
          bool Closable>
static void testWaitingPolicies() {
  using Queue =
      rigtorp::SPSCQueue<int, WaitingConfig<Producer, Consumer, Closable>>;
  constexpr int transfers = 10000;
  for (size_t capacity : {size_t{1}, size_t{3}}) {
    Queue q(capacity);
    std::thread producer([&] {
      for (int i = 0; i < transfers; ++i) {
        q.emplace(i);
      }
      if constexpr (Closable) {
        q.complete();
      }
    });
    for (int i = 0; i < transfers; ++i) {
      if (i % 3 == 0) {
        auto *item = q.wait_front();
        CHECK(item && *item == i);
        q.pop();
      } else if (i % 3 == 1) {
        int value = -1;
        if constexpr (Closable) {
          CHECK(q.wait_pop(value));
        } else {
          q.wait_pop(value);
        }
        CHECK(value == i);
      } else {
        auto checkValue = [i](int &value) { CHECK(value == i); };
        if constexpr (Closable) {
          CHECK(q.consume(checkValue));
        } else {
          q.consume(checkValue);
        }
      }
    }
    if constexpr (Closable) {
      CHECK(q.wait_front() == nullptr);
      int value = -1;
      CHECK(!q.wait_pop(value));
      CHECK(value == -1);
      CHECK(!q.consume([](int &) { CHECK(false); }));
      CHECK(q.completed());
    }
    producer.join();
    CHECK(q.empty());
  }
}

static void testAtomicWakeups() {
  using namespace std::chrono_literals;
  using Queue = constraints_test::ClosableQueue;
  using Receive = void (*)(Queue &, int);

  auto waitFront = [](Queue &q, int expected) {
    auto *item = q.wait_front();
    if (expected == 0) {
      CHECK(item == nullptr);
    } else {
      CHECK(item && *item == expected);
      q.pop();
    }
  };
  auto waitPop = [](Queue &q, int expected) {
    int value = -1;
    CHECK(q.wait_pop(value) == (expected != 0));
    CHECK(value == (expected == 0 ? -1 : expected));
  };
  auto consume = [](Queue &q, int expected) {
    int value = -1;
    CHECK(q.consume([&value](int &item) { value = item; }) == (expected != 0));
    CHECK(value == (expected == 0 ? -1 : expected));
  };

  for (auto receive : std::array<Receive, 3>{waitFront, waitPop, consume}) {
    for (int value : {0, 1, 2}) {
      Queue q(1);
      std::thread consumer([&] { receive(q, value); });
      std::this_thread::sleep_for(5ms);
      if (value == 0) {
        q.complete();
      } else {
        CHECK(value == 1 ? q.try_push(value) : q.try_emplace(value));
      }
      consumer.join();
    }
  }

  auto tryPop = [](Queue &q, int expected) {
    int value = 0;
    CHECK(q.try_pop(value) && value == expected);
  };
  auto tryConsume = [](Queue &q, int expected) {
    CHECK(q.try_consume([expected](int &value) { CHECK(value == expected); }));
  };
  for (auto receive :
       std::array<Receive, 4>{tryConsume, tryPop, consume, waitPop}) {
    Queue q(1);
    q.push(1);
    std::thread producer([&] { q.push(2); });
    std::this_thread::sleep_for(5ms);
    receive(q, 1);
    producer.join();
    CHECK(*q.wait_front() == 2);
    q.pop();
  }
}

static void testCloseAndDrain() {
  using rigtorp::WaitPolicy;
  using Queue = rigtorp::SPSCQueue<
      int, WaitingConfig<WaitPolicy::Spin, WaitPolicy::Atomic, true>>;
  std::array<Queue, 2> queues{Queue(2), Queue(2)};
  for (auto &q : queues) {
    CHECK(q.try_push(1));
    CHECK(!q.completed());
  }
  CHECK(queues[0].try_push(2));
  CHECK(!queues[0].try_push(3));
  queues[0].complete(); // Abandon the full queue.
  auto drain = [](Queue &q, int count) {
    auto *item = q.front();
    CHECK(item && *item == 1);
    CHECK(q.wait_front() == item);
    q.pop();
    int value = 0;
    if (count == 2) {
      CHECK(q.try_pop(value) && value == 2);
    }
    CHECK(q.wait_front() == nullptr);
    CHECK(q.completed() && q.empty() && q.size() == 0);
    CHECK(q.front() == nullptr);
    CHECK(!q.try_pop(value));
    CHECK(!q.try_consume([](int &) { CHECK(false); }));
  };
  CHECK(queues[0].completed() && queues[0].size() == 2);
  std::thread full([&] { drain(queues[0], 2); });
  std::thread partial([&] { drain(queues[1], 1); });
  queues[1].complete(); // Graceful shutdown of the remaining queue.
  queues[1].complete();
  full.join();
  partial.join();

  Queue q(2);
  CHECK(q.try_push(1));
  CHECK(q.try_push(2));
  q.complete();
  int value = 0;
  CHECK(q.wait_pop(value) && value == 1);
  CHECK(q.consume([](int &item) { CHECK(item == 2); }));
  CHECK(!q.wait_pop(value) && value == 1);
  CHECK(!q.consume([](int &) { CHECK(false); }));
}

static void testDeathPopWithoutFront(bool enqueue) {
#if (defined(__unix__) || defined(__APPLE__)) && !defined(NDEBUG)
  using namespace rigtorp;

  std::cout.flush();
  std::cerr.flush();

  pid_t pid = fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    // Exit from the SIGABRT handler so assert() failure does not invoke
    // coredumpctl / ABRT via default SIGABRT termination.
    std::signal(SIGABRT, [](int) { _exit(42); });
    SPSCQueue<int> q(2);
    if (enqueue) {
      q.push(1);
    }
    q.pop(); // Must abort in debug builds without a successful front()
    _exit(0);
  }

  int status = 0;
  CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 42);
#else
  (void)enqueue;
#endif
}

static void testConsumerOperations() {
  using namespace rigtorp;

  // Empty, wrap-around, and cached writeIdx update
  {
    SPSCQueue<int> q(2);
    int v = -1;
    CHECK(!q.try_pop(v));
    CHECK(v == -1);

    for (int i = 0; i < 6; ++i) {
      CHECK(q.try_push(i));
      CHECK(q.try_pop(v));
      CHECK(v == i);
      CHECK(!q.try_pop(v));
    }
  }

  // Move-only type
  {
    SPSCQueue<std::unique_ptr<int>> q(2);
    q.emplace(std::make_unique<int>(99));
    std::unique_ptr<int> out;
    CHECK(q.try_pop(out));
    CHECK(out && *out == 99);
    CHECK(!q.try_pop(out));
    q.emplace(std::make_unique<int>(100));
    static_assert(noexcept(q.wait_pop(out)));
    q.wait_pop(out);
    CHECK(out && *out == 100);
    CHECK(q.empty());
  }

  // Copy-only type
  {
    struct CopyOnly {
      int x;
      explicit CopyOnly(int v) noexcept : x(v) {}
      CopyOnly(const CopyOnly &) = default;
      CopyOnly &operator=(const CopyOnly &) = default;
      CopyOnly(CopyOnly &&) = delete;
      CopyOnly &operator=(CopyOnly &&) = delete;
    };
    SPSCQueue<CopyOnly> q(2);
    CopyOnly in(7);
    q.push(in);
    CopyOnly out(0);
    CHECK(q.try_pop(out));
    CHECK(out.x == 7);
    CHECK(!q.try_pop(out));
    q.emplace(8);
    static_assert(noexcept(q.wait_pop(out)));
    q.wait_pop(out);
    CHECK(out.x == 8);
    CHECK(q.empty());
  }

  // A throwing assignment leaves the item available for retry.
  {
    struct Item {
      int value;
      bool shouldThrow;
      Item &operator=(Item &&source) {
        value = source.value;
        if (shouldThrow) {
          ++source.value;
          throw std::runtime_error("assignment failed");
        }
        return *this;
      }
    };
    SPSCQueue<Item> q(1);
    Item out{0, true};
    static_assert(!noexcept(q.wait_pop(out)));
    q.emplace(42, false);
    try {
      q.wait_pop(out);
      CHECK(false);
    } catch (const std::runtime_error &) {
      CHECK(out.value == 42);
      auto *item = q.front();
      CHECK(item && item->value == 43);
    }
    out.shouldThrow = false;
    q.wait_pop(out);
    CHECK(out.value == 43);
    CHECK(q.empty());
  }

  // Member-function callbacks use the same invocation rules as callable
  // objects.
  {
    struct Item {
      int *calls;
      void consume() noexcept { ++*calls; }
      void fail() { throw std::runtime_error("callback failed"); }
    };
    SPSCQueue<Item> q(1);
    static_assert(noexcept(q.try_consume(&Item::consume)));
    static_assert(!noexcept(q.try_consume(&Item::fail)));
    int calls = 0;
    CHECK(!q.try_consume(&Item::consume));
    q.emplace(&calls);
    try {
      (void)q.try_consume(&Item::fail);
      CHECK(false);
    } catch (const std::runtime_error &) {
      CHECK(q.front() != nullptr);
    }
    CHECK(q.try_consume(&Item::consume));
    CHECK(calls == 1);
    CHECK(q.empty());
    static_assert(noexcept(q.consume(&Item::consume)));
    static_assert(!noexcept(q.consume(&Item::fail)));
    q.emplace(&calls);
    try {
      q.consume(&Item::fail);
      CHECK(false);
    } catch (const std::runtime_error &) {
      CHECK(q.front() != nullptr);
    }
    q.consume(&Item::consume);
    CHECK(calls == 2);
    CHECK(q.empty());
  }

  // In-place consumption with immovable and move-only types
  {
    struct Immovable {
      int x;
      explicit Immovable(int v) noexcept : x(v) {}
      Immovable(const Immovable &) = delete;
      Immovable &operator=(const Immovable &) = delete;
      Immovable(Immovable &&) = delete;
      Immovable &operator=(Immovable &&) = delete;
    };
    struct Callback {
      int &observed;
      int operator()(Immovable &item) && noexcept {
        observed = item.x;
        return item.x;
      }
    };
    SPSCQueue<Immovable> q(2);
    int observed = 0;
    CHECK(!q.try_consume([&](Immovable &v) { observed = v.x; }));
    q.emplace(123);
    CHECK(q.try_consume([&](Immovable &v) { observed = v.x; }));
    CHECK(observed == 123);
    CHECK(q.empty());
    q.emplace(321);
    q.consume(Callback{observed});
    CHECK(observed == 321);
    CHECK(q.empty());

    SPSCQueue<std::unique_ptr<int>> mq(2);
    mq.emplace(std::make_unique<int>(456));
    std::unique_ptr<int> out;
    CHECK(mq.try_consume([&](std::unique_ptr<int> &p) { out = std::move(p); }));
    CHECK(out && *out == 456);
    CHECK(mq.empty());
  }
}

int main(int argc, char *argv[]) {
  (void)argc, (void)argv;

  testFunctionality();
  testTypeCategories();
  testExceptionSafety();
  testCustomAllocator();
  testCapacityBounds();
  testAlignedElementPadding();
  testCapacityAllocationBounds<RecordingAllocator<char>>();
  testCapacityAllocationBounds<RecordingAtLeastAllocator<char>>();
  testCapacityAllocationBounds<RecordingAllocator<char>, true>();
  testCapacityAllocationBounds<RecordingAtLeastAllocator<char>, true>();
  testDeathPopWithoutFront(false);
  testDeathPopWithoutFront(true);
  std::promise<void> finished;
  std::thread watchdog([done = finished.get_future()] {
    CHECK(done.wait_for(std::chrono::seconds(60)) == std::future_status::ready);
  });
  testConsumerOperations();
  testConcurrency();
  using rigtorp::WaitPolicy;
  testWaitingPolicies<WaitPolicy::Spin, WaitPolicy::Spin, false>();
  testWaitingPolicies<WaitPolicy::Spin, WaitPolicy::Atomic, false>();
  testWaitingPolicies<WaitPolicy::Atomic, WaitPolicy::Spin, false>();
  testWaitingPolicies<WaitPolicy::Atomic, WaitPolicy::Atomic, false>();
  testWaitingPolicies<WaitPolicy::Spin, WaitPolicy::Spin, true>();
  testWaitingPolicies<WaitPolicy::Spin, WaitPolicy::Atomic, true>();
  testWaitingPolicies<WaitPolicy::Atomic, WaitPolicy::Spin, true>();
  testWaitingPolicies<WaitPolicy::Atomic, WaitPolicy::Atomic, true>();
  testAtomicWakeups();
  testCloseAndDrain();
  finished.set_value();
  watchdog.join();

  return 0;
}
