#include <iostream>

#ifdef __AVX512F__
void id()
{
    printf("512");
}
#elif defined(__AVX2__)
void id()
{
    printf("AVX2");
}
#else
void id()
{
    printf("generic");
}
#endif
