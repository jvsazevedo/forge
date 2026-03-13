namespace Forge

open Core.Lib

module Fibo =
    let run (n: int64) =
        printfn "  fibonacci(%2d) = %d" n (rust_fibonacci n)