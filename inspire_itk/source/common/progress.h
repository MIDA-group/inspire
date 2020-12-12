
#ifndef PROGRESS_H
#define PROGRESS_H

#include <iostream>
#include <mutex>

class Progress {
public:
    Progress(size_t taskCount, size_t barElements) : m_TaskCount(taskCount), m_Completed(0), m_BarElements(barElements) {
        ;
    }
    ~Progress() = default;
    Progress(const Progress&) = delete;
    Progress(Progress&&) = delete;

    Progress &operator=(const Progress&) = delete;
    Progress &operator=(Progress&&) = delete;

    void StartProgressUpdate() {
        m_Lock.lock();
    }

    template <typename TStream>
    void ReportProgressUpdate(TStream& strm, bool remove=false) {
        assert(m_Completed < m_TaskCount);
        ++m_Completed;

        double fraction = (double)m_Completed / (double)m_TaskCount;
        size_t filledElements = (size_t)(fraction * m_BarElements);
        std::string buf;
        buf.append("[");
        for(size_t i = 0; i < m_BarElements; ++i) {
            if (i < filledElements) {
                buf.append("=");
            } else if (i == filledElements) {
                buf.append(">");
            } else {
                buf.append(" ");
            }
        }
        buf.append("]");

        char buf2[512];
        sprintf(buf2, " %.2f%%", 100.0 * fraction);
 
        buf.append(buf2);

        if(remove) {
            buf.append("\r");
        } else {
            buf.append("\n");
        }

        strm << buf.c_str();
        strm.flush();
    }

    void EndProgressUpdate() {
        m_Lock.unlock();
    }
private:
    size_t m_TaskCount;
    size_t m_Completed;
    size_t m_BarElements;

    std::mutex m_Lock;
};

#endif
