module Tests

open System
open Xunit

open Forge.Core.Lib

[<Fact>]
let ``My test`` () =
    let n = 10L
    let result =
        rust_fibonacci n

    printfn "called from rust fibonacci(%2d) = %d" n result
    Assert.True(true)
