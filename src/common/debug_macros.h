#ifndef DEBUG_MACROS_H
#define DEBUG_MACROS_H

#ifdef DEBUG_MODE
    #define DEBUG_PRINTF(...) fprintf(stderr, __VA_ARGS__)
    #define DEBUG_LOG(msg) fprintf(stderr, "[DEBUG] %s\n", msg)
    #define DEBUG_PRINT_LUM_STATE(lum) print_lum_debug_state(lum)
#else
    #define DEBUG_PRINTF(...) ((void)0)
    #define DEBUG_LOG(msg) ((void)0)
    #define DEBUG_PRINT_LUM_STATE(lum) ((void)0)
#endif

#endif // DEBUG_MACROS_H