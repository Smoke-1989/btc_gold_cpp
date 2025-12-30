#include "logger.h"

namespace btc_gold {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

}  // namespace btc_gold
