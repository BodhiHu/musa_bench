# musa bench

> 🚧🚧🚧 under construction ...

#### dp4a

> `-fsigned-char` ensures that all char variables behave as signed char,
> meaning they can store values from -128 to 127 (on 8-bit char systems).
> 
> Without this flag, the default behavior varies:
>  - On x86 (Intel/AMD): char is usually signed by default.
>  - On ARM (including ARM64/NEON): char is often unsigned by default.

```
mcc bench_dp4a.mu -lmusart -fsigned-char -o bench_dp4a
./bench_dp4a
```
