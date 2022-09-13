#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <sys/wait.h>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using ::testing::HasSubstr;

const auto max_assertions_failure_str = "Assertion failure " +
    std::to_string(C10_CUDA_DEVICE_SIDE_ASSERTION_COUNT - 1);

/**
 * Device kernel that takes a single integer parameter as argument and
 * will always trigger a device side assertion.
 */
__global__ void cuda_always_fail_assertion_kernel(
    const int a,
    CUDA_KERNEL_ASSERT_ARGS) {
  CUDA_KERNEL_ASSERT2(a != a);
}

/**
 * Device kernel that takes a single integer parameter as argument and
 * will never trigger a device side assertion.
 */
__global__ void cuda_always_succeed_assertion_kernel(
    const int a,
    CUDA_KERNEL_ASSERT_ARGS) {
  CUDA_KERNEL_ASSERT2(a == a);
}

/**
 * Device kernel that takes a single clock_t parameter as argument,
 * @param clock_count represents the time each executing function will wait
 * before triggering an assertion. The kernel will always trigger a device side
 * assertion.
 */
__global__ void cuda_wait_a_bit_then_fail_kernel(
    const clock_t clock_count,
    CUDA_KERNEL_ASSERT_ARGS) {
  const auto start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < clock_count) {
    clock_offset = clock() - start_clock;
  }
  CUDA_KERNEL_ASSERT2(false);
}

/**
 * Device kernel that takes mulitple integer parameters as arguments and
 * will always trigger a device side assertion.
 */
__global__ void cuda_multiple_vars_always_fail_assertion_kernel(
    const int a,
    const int b,
    const int c,
    const int d,
    CUDA_KERNEL_ASSERT_ARGS) {
  int i = a + b + c + d;
  if (i != 0) {
    CUDA_KERNEL_ASSERT2(i == -i);
  } else {
    CUDA_KERNEL_ASSERT2(i == i + 1);
  }
}

/**
 * Device kernel that takes 2 arguments
 * @param bad_thread represents the thread we want to trigger assertion on.
 * @param bad_block represents the block we want to trigger assertion on.
 * This kernel will only trigger a device side assertion for <<bad_block,
 * bad_thread>> pair. all the other blocks and threads pairs will basically be
 * no-op.
 */
__global__ void cuda_device_assertions_fail_on_thread_block_kernel(
    const int bad_thread,
    const int bad_block,
    CUDA_KERNEL_ASSERT_ARGS) {
  if (threadIdx.x == bad_thread && blockIdx.x == bad_block) {
    CUDA_KERNEL_ASSERT2(false); // This comparison necessarily needs to fail
  }
}

/**
 * Helper methods, for beautifying the logs in console.
 */
void running_msg(std::string function_name) {
  std::cerr << "\n\n\033[94m### Running " << function_name << "...\033[39m"
            << std::endl;
}

void success_msg(std::string function_name) {
  std::cerr << "\033[92m### " << function_name << " succeeded!\033[39m"
            << std::endl;
}

/**
 * TEST: Triggering device side assertion on a simple <<<1,1>>> config.
 * kernel used takes only 1 variable as parameter function.
 */
void cuda_device_assertions_1_var_test() {
  running_msg(__FUNCTION__);
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str,
        HasSubstr("CUDA device-side assertion failures were found on GPU #0!"));
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion on a simple <<<1,1>>> config.
 * kernel used takes multiple variables as parameters to the function.
 */
void cuda_device_assertions_catches_stream() {
  running_msg(__FUNCTION__);
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_KERNEL_LAUNCH(
      cuda_multiple_vars_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1, /* const int a */
      2, /* const int b */
      3, /* const int c */
      4 /* const int d */
  );

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str, HasSubstr("# of GPUs this process interacted with = 1"));
    ASSERT_THAT(
        err_str,
        HasSubstr("CUDA device-side assertion failures were found on GPU #0!"));
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_multiple_vars_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion on only 1 thread from <<<1024,128>>>
 * grid. kernel used is unique, it take 2 parameters to tell which particular
 * block and thread it should assert, all the other theads of the kernel will be
 * basically no-op.
 */
void cuda_device_assertions_catches_thread_and_block_and_device() {
  running_msg(__FUNCTION__);
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_KERNEL_LAUNCH(
      cuda_device_assertions_fail_on_thread_block_kernel,
      1024, /* Blocks */
      128, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      29, /* bad thread */
      937 /* bad block */
  );

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [29,0,0]"));
    ASSERT_THAT(
        err_str, HasSubstr("Block ID that failed assertion = [937,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_device_assertions_fail_on_thread_block_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion from single block and multiple threads
 * <<<1,128>>>. Once the very first thread asserts all the other threads will
 * basically be in bad state and the block id with failed asseriton would be
 * [0,0,0].
 */
void cuda_device_assertions_multiple_writes_from_same_block() {
  running_msg(__FUNCTION__);
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      1, /* Blocks */
      128, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(err_str, HasSubstr(max_assertions_failure_str));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion from multiple block but single thread
 * <<<10,1>>>. Here we are triggering assertion on 10 blocks, each with only 1
 * thread. Since we have more than 10 SM on a GPU, we expect each block to be
 * executed and successfully assert, Hence we will see assertions logged from
 * each block here.
 */
void cuda_device_assertions_multiple_writes_from_multiple_blocks() {
  running_msg(__FUNCTION__);
  const auto stream = c10::cuda::getStreamFromPool();
  TORCH_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      10, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream, /* Stream */
      1);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(err_str, HasSubstr(max_assertions_failure_str));
    ASSERT_THAT(
        err_str, HasSubstr("Thread ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [0,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [1,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [2,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [3,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [4,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [5,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [6,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [7,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [8,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Block ID that failed assertion = [9,0,0]"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream.id())));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion from multiple block but single thread
 * <<<10,128>>>. Here we are triggering assertion on 10 blocks, each with only
 * 128 thread.
 */
void cuda_device_assertions_multiple_writes_from_blocks_and_threads() {
  running_msg(__FUNCTION__);
  bool run_threads = false;

  // Create a function to launch kernel that waits for a signal, to try to
  // ensure everything is happening simultaneously
  const auto launch_the_kernel = [&]() {
    // Busy loop waiting for the signal to go
    while (!run_threads) {
    }

    TORCH_KERNEL_LAUNCH(
        cuda_always_fail_assertion_kernel,
        10, /* Blocks */
        128, /* Threads */
        0, /* Shared mem */
        c10::cuda::getCurrentCUDAStream(), /* Stream */
        1);
  };

  // Spin up a bunch of busy-looping threads
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; i++) {
    threads.emplace_back(launch_the_kernel);
  }

  // Paranoid - wait for all the threads to get setup
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Mash
  run_threads = true;

  // Clean-up
  for (auto& x : threads) {
    x.join();
  }

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(err_str, HasSubstr(max_assertions_failure_str));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion from 2 streams.
 * We intentionally put one stream to sleep while the other one asserts, to make
 * sure that one stream does not interfere with other streams assertions.
 * TODO: This test is flakey.
 */
void cuda_device_assertions_multiple_writes_from_2_streams() {
  running_msg(__FUNCTION__);
  const auto stream1 = c10::cuda::getStreamFromPool();
  const auto stream2 = c10::cuda::getStreamFromPool();

  ASSERT_NE(stream1.id(), stream2.id());

  constexpr auto seconds = 1;
  // An upper limit
  constexpr int64_t clock_speed = 2'100'000'000;
  constexpr auto wait_for1 = seconds * clock_speed / 2;
  constexpr auto wait_for2 = seconds * clock_speed;

  TORCH_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream1,
      wait_for1);

  TORCH_KERNEL_LAUNCH(
      cuda_always_fail_assertion_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      stream2,
      wait_for2);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "2 CUDA device-side assertion failures were found on GPU #0!"));
    ASSERT_THAT(err_str, HasSubstr("Assertion failure 0"));
    ASSERT_THAT(err_str, HasSubstr("Assertion failure 1"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream1.id())));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Stream kernel was launched on = " + std::to_string(stream2.id())));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Name of kernel launched that led to failure = cuda_always_fail_assertion_kernel"));
    ASSERT_THAT(
        err_str, HasSubstr("File containing kernel launch = " __FILE__));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "Function containing kernel launch = " +
            std::string(__FUNCTION__)));
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertion from 2 different processes from CPU.
 * The following code is testing if two processes from CPU that are running
 * GPU kernels (not necessarily simultaneously) and are asserting & writing
 * to the respective UVMs, mess up anything for each other.
 * Once parent process's kernel launch fails and causes a device-side assertion
 * and is still alive when the second process is interacting with the GPU,
 * trying to launch another kernel.
 */
void cuda_device_assertions_from_2_processes() {
  running_msg(__FUNCTION__);

  const auto n1 = fork();
  if (n1 == 0) {
    // This is the parent process, that will call an assertion failure.
    // This should execute before the child process.
    // We are achieving this by putting the child process to sleep.
    TORCH_KERNEL_LAUNCH(
        cuda_always_fail_assertion_kernel,
        1, /* Blocks */
        1, /* Threads */
        0, /* Shared mem */
        c10::cuda::getStreamFromPool(), /* Stream */
        1);
    try {
      c10::cuda::device_synchronize();
      throw std::runtime_error("Test didn't fail, but should have.");
    } catch (const c10::Error& err) {
      const auto err_str = std::string(err.what());
      ASSERT_THAT(
          err_str,
          HasSubstr(
              "1 CUDA device-side assertion failures were found on GPU #0!"));
    }
    // Keep this alive so we can see what happened to the other process
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
  } else {
    // This is the child process
    // We put it to sleep for next 2 seconds, to make sure that the parent has
    // asserted a failure already.
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    TORCH_KERNEL_LAUNCH(
        cuda_always_succeed_assertion_kernel,
        1, /* Blocks */
        1, /* Threads */
        0, /* Shared mem */
        c10::cuda::getStreamFromPool(), /* Stream */
        1);
    try {
      c10::cuda::device_synchronize();
    } catch (const c10::Error& err) {
      ASSERT_TRUE(false); // This kernel should not have failed, but did.
    }
    // End the child process
    exit(0);
  }

  success_msg(__FUNCTION__);
}

/**
 * TEST: Triggering device side assertions on kernels that are running on two
 * different GPUs on same machine. Once a CUDA error is noticed PyTorch throws
 * an exception our last chance to get info about what happened on the GPUs is
 * at that time, so we want to ensure we can see info from any and all GPUs this
 * process may have been working with.
 * TODO: This test is flakey.
 */
void cuda_device_assertions_on_multiple_gpus() {
  running_msg(__FUNCTION__);

  int device_count;
  C10_CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count < 2) {
    success_msg(__FUNCTION__);
  }

  constexpr auto seconds = 1;
  // An upper limit
  constexpr int64_t clock_speed = 2'100'000'000;
  constexpr auto wait_for1 = seconds * clock_speed / 2;
  constexpr auto wait_for2 = seconds * clock_speed;

  // Okay, we have 2 devices. Let's use them to launch some kernels
  c10::cuda::set_device(0);
  TORCH_KERNEL_LAUNCH(
      cuda_wait_a_bit_then_fail_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      c10::cuda::getStreamFromPool(),
      wait_for1);

  c10::cuda::set_device(1);
  TORCH_KERNEL_LAUNCH(
      cuda_wait_a_bit_then_fail_kernel,
      1, /* Blocks */
      1, /* Threads */
      0, /* Shared mem */
      c10::cuda::getStreamFromPool(),
      wait_for2);

  try {
    c10::cuda::device_synchronize();
    throw std::runtime_error("Test didn't fail, but should have.");
  } catch (const c10::Error& err) {
    const auto err_str = std::string(err.what());
    ASSERT_THAT(
        err_str, HasSubstr("# of GPUs this process interacted with = 2"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "1 CUDA device-side assertion failures were found on GPU #0!"));
    ASSERT_THAT(
        err_str,
        HasSubstr(
            "1 CUDA device-side assertion failures were found on GPU #1!"));
    ASSERT_THAT(err_str, HasSubstr("Assertion failure 0"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 0"));
    ASSERT_THAT(err_str, HasSubstr("Device that launched kernel = 1"));
  }

  success_msg(__FUNCTION__);
}

int main() {
  const std::vector<std::function<void(void)>> test_list = {
      cuda_device_assertions_1_var_test,
      cuda_device_assertions_catches_stream,
      cuda_device_assertions_catches_thread_and_block_and_device,
      cuda_device_assertions_multiple_writes_from_same_block,
      cuda_device_assertions_multiple_writes_from_multiple_blocks,
      cuda_device_assertions_multiple_writes_from_blocks_and_threads,
      cuda_device_assertions_from_2_processes
      // TODO: these tests seem to be flakey
      // cuda_device_assertions_multiple_writes_from_2_streams,
      // cuda_device_assertions_on_multiple_gpus
      /*
       * Learnings with these flakey tests in local testing have been:
       * - that an issue on one stream seems to make the other streams
       * inaccessible and prevents them from asserting.
       * - that the same seems true of multiple GPUs
       * - that in both cases if it turned out that multiple streams or multiple
       * GPUs could assert at the same time, the code would handle this just
       * fine.
       */
  };

  // In order to recover the GPU after an assertion failure, we need to kill the
  // process that launched the kernel that caused the failure. In order to make
  // this compatible with running a series of tests, we use fork at the
  // beginning of each test and run the test in the child process. See:
  // https://stackoverflow.com/a/56330491
  int successful_tests = 0;
  for (const auto& test : test_list) {
    const auto pid = fork();
    if (pid == 0) { // Child process
      test();
      // If child process throws an exception we should not make it to this line
      exit(0);
    } else { // Parent process
      int status;
      // Wait for the child process to finish so only one test is running on the
      // GPU at a time
      waitpid(pid, &status, 0);
      successful_tests += (status == 0);
    }
  }

  std::cerr << "Successful tests = " << successful_tests << " of "
            << test_list.size() << std::endl;
}
