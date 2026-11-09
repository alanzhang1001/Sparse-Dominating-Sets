// Linux process measurement using the kernel's per-child peak RSS (wait4).
// The lightweight parent is excluded; no sampling interval can miss a peak.
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

static volatile sig_atomic_t child_pid = -1;
static volatile sig_atomic_t timed_out = 0;
static void deadline(int) {
    timed_out = 1;
    if (child_pid > 0) kill(child_pid, SIGKILL);
}
int main(int argc, char** argv) {
    if (argc < 5) return 2; // TIMEOUT STDOUT STDERR PROGRAM [ARGS...]
    const double timeout = std::stod(argv[1]);
    if (!(timeout > 0 && timeout <= 86400)) return 2;
    const int out = open(argv[2], O_CREAT|O_TRUNC|O_WRONLY, 0644);
    const int err = open(argv[3], O_CREAT|O_TRUNC|O_WRONLY, 0644);
    if (out < 0 || err < 0) return 2;
    struct sigaction handler{};
    handler.sa_handler = deadline;
    sigemptyset(&handler.sa_mask);
    sigaction(SIGALRM, &handler, nullptr);
    const auto started = std::chrono::steady_clock::now();
    const pid_t pid = fork();
    if (pid == 0) {
        if (dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) _exit(126);
        close(out); close(err);
        execvp(argv[4], argv+4);
        _exit(127);
    }
    close(out); close(err);
    if (pid < 0) return 2;
    child_pid = pid;
    alarm(static_cast<unsigned>(std::ceil(timeout)));
    int status = 0; struct rusage usage{};
    pid_t waited;
    do { waited = wait4(pid, &status, 0, &usage); } while (waited < 0 && errno == EINTR);
    alarm(0); child_pid = -1;
    if (waited < 0) return 2;
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128+WTERMSIG(status);
    std::cout << std::setprecision(17) << "{\"process_wall_seconds\":" << seconds
              << ",\"peak_rss_kib\":" << usage.ru_maxrss
              << ",\"returncode\":" << code << ",\"timed_out\":" << (timed_out ? "true" : "false") << "}\n";
    return 0;
}
