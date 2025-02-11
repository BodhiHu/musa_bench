#include <musa.h>
#include <stdio.h>

__global__ void dp4a_test_kernel(int *ret, int a, int b, int c) {

    int dp4a_ret = 0, ref_ret = 0;

    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;

    printf("\ndp4a >>\n");
    printf("calc: %d + (%d * %d) + (%d * %d) + (%d * %d) + (%d * %d)\n",
        c, a8[0], b8[0], a8[1], b8[1], a8[2], b8[2], a8[3], b8[3]
    );

    ref_ret = c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
    dp4a_ret = __dp4a(a, b, c);
    int delta = dp4a_ret - ref_ret;

    printf("ref_ret = %6d, dp4a_ret = %6d, delta = %d\n",
        ref_ret, dp4a_ret, delta
    );

    *ret = delta;
}

int test_dp4a() {
    int h_output = -1;
    int *d_output;
    int c = 100;
    int idx = 0;

    // Allocate device memory
    musaMalloc((void**)&d_output, sizeof(int));

    // packed 8-bit integers (4 values per int) example:
    // 0x01020304: A = [4, 3, 2, 1] (LSB to MSB)
    // 0x05060708: B = [8, 7, 6, 5] (LSB to MSB)

    // Launch kernel (1 block, 1 thread)
    dp4a_test_kernel<<<1, 1>>>(d_output, 0x01020304, 0x05060708, ++c);
    // Copy result back to host
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x11121314, 0x15161718, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x31131374, 0x35767713, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x7f797477, 0x78787878, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x7f797477, 0x65769713, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x8f897477, 0x65769713, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0x8f898487, 0xa5b6c7d3, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0xf1f3f3f4, 0x35769713, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0xf1f3f3f4, 0xf5f6f7f3, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    dp4a_test_kernel<<<1, 1>>>(d_output, 0xf1f3f3f4, 0x00000000, ++c);
    musaMemcpy(&h_output, d_output, sizeof(int), musaMemcpyDeviceToHost);
    printf(">> Dot Product (dp4a) test %d pass: %s\n", idx++, h_output == 0 ? "true" : "false");

    // Free memory
    musaFree(d_output);

    return 0;
}

int main() {
    return test_dp4a();
}
