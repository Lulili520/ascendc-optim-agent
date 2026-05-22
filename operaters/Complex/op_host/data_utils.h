// Data I/O utilities (copied from template)
#ifndef DATA_UTILS_H
#define DATA_UTILS_H

#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define ERROR_LOG(fmt, ...) \
    do { printf("[ERROR] " fmt "\n", ##__VA_ARGS__); } while (0)

inline bool ReadFile(const char* filePath, uint32_t bufferSize, void* buffer, uint32_t bufferLen)
{
    if (buffer == nullptr) {
        ERROR_LOG("buffer is nullptr");
        return false;
    }
    if (bufferSize > bufferLen) {
        ERROR_LOG("bufferSize %u > bufferLen %u", bufferSize, bufferLen);
        return false;
    }
    int fd = open(filePath, O_RDONLY);
    if (fd == -1) {
        ERROR_LOG("open file %s failed", filePath);
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) == -1) {
        ERROR_LOG("fstat file %s failed", filePath);
        close(fd);
        return false;
    }
    if ((uint32_t)st.st_size != bufferSize) {
        ERROR_LOG("file size %ld != bufferSize %u", (long)st.st_size, bufferSize);
        close(fd);
        return false;
    }
    ssize_t ret = read(fd, buffer, bufferSize);
    close(fd);
    if (ret == -1) {
        ERROR_LOG("read file %s failed", filePath);
        return false;
    }
    return true;
}

inline bool WriteFile(const char* filePath, void* buffer, uint32_t size)
{
    int fd = open(filePath, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        ERROR_LOG("open file %s failed for write", filePath);
        return false;
    }
    ssize_t ret = write(fd, buffer, size);
    close(fd);
    if (ret == -1) {
        ERROR_LOG("write file %s failed", filePath);
        return false;
    }
    return true;
}

#endif // DATA_UTILS_H
