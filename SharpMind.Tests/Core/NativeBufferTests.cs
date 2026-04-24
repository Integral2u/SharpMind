using SharpMind.Core.Memory;

namespace SharpMind.Tests.Core
{
    public sealed class NativeBufferTests
    {
        [Fact]
        public void Allocate_ZeroInitialised()
        {
            using var buf = new NativeBuffer<float>(64);
            foreach (float v in buf.AsSpan())
                Assert.Equal(0f, v);
        }

        [Fact]
        public void AddRef_PreventsFreeUntilAllDisposed()
        {
            var buf = new NativeBuffer<float>(8);
            buf.AddRef();               // ref = 2
            buf.Dispose();              // ref = 1 — should NOT free yet
            buf[0] = 42f;               // would fault if freed
            Assert.Equal(42f, buf[0]);
            buf.Dispose();              // ref = 0 — frees now
        }

        [Fact]
        public void Dispose_AfterFree_DoesNotThrow()
        {
            var buf = new NativeBuffer<float>(4);
            buf.Dispose();
            var ex = Record.Exception(() => buf.Dispose());
            Assert.Null(ex);            // double-dispose is safe (ref hits 0 once)
        }
    }
}
