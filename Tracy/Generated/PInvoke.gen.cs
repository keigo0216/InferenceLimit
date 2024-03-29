
// <auto-generated>
//  This code was generated by the following tool on 2023-10-21 13:11:21 GMT-04:00:
//      https://github.com/bottlenoselabs/c2cs (v6.1.3.0)
//
//  Changes to this file may cause incorrect behavior and will be lost if the code is regenerated.
// </auto-generated>
// ReSharper disable All

#region Template
#nullable enable
#pragma warning disable CS1591
#pragma warning disable CS8981
using bottlenoselabs.C2CS.Runtime;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
#endregion

namespace Tracy;

public static unsafe class PInvoke
{
    private const string LibraryName = "TracyClient";

    #region API

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_alloc_srcloc")]
    public static extern ulong TracyAllocSrcloc(uint line, CString source, ulong sourceSz, CString function, ulong functionSz);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_alloc_srcloc_name")]
    public static extern ulong TracyAllocSrclocName(uint line, CString source, ulong sourceSz, CString function, ulong functionSz, CString name, ulong nameSz);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_connected")]
    public static extern int TracyConnected();

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_frame_image")]
    public static extern void TracyEmitFrameImage(void* image, ushort w, ushort h, byte offset, int flip);

    [DllImport(LibraryName, EntryPoint = "___tracy_emit_frame_mark", CallingConvention = CallingConvention.Cdecl)]
    public static extern void TracyEmitFrameMark(CString name);

    [DllImport(LibraryName, EntryPoint = "___tracy_emit_frame_mark_end", CallingConvention = CallingConvention.Cdecl)]
    public static extern void TracyEmitFrameMarkEnd(CString name);

    [DllImport(LibraryName, EntryPoint = "___tracy_emit_frame_mark_start", CallingConvention = CallingConvention.Cdecl)]
    public static extern void TracyEmitFrameMarkStart(CString name);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_calibration")]
    public static extern void TracyEmitGpuCalibration(TracyGpuCalibrationData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_calibration_serial")]
    public static extern void TracyEmitGpuCalibrationSerial(TracyGpuCalibrationData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_context_name")]
    public static extern void TracyEmitGpuContextName(TracyGpuContextNameData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_context_name_serial")]
    public static extern void TracyEmitGpuContextNameSerial(TracyGpuContextNameData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_new_context")]
    public static extern void TracyEmitGpuNewContext(TracyGpuNewContextData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_new_context_serial")]
    public static extern void TracyEmitGpuNewContextSerial(TracyGpuNewContextData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_time")]
    public static extern void TracyEmitGpuTime(TracyGpuTimeData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_time_serial")]
    public static extern void TracyEmitGpuTimeSerial(TracyGpuTimeData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin")]
    public static extern void TracyEmitGpuZoneBegin(TracyGpuZoneBeginData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_alloc")]
    public static extern void TracyEmitGpuZoneBeginAlloc(TracyGpuZoneBeginData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_alloc_callstack")]
    public static extern void TracyEmitGpuZoneBeginAllocCallstack(TracyGpuZoneBeginCallstackData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_alloc_callstack_serial")]
    public static extern void TracyEmitGpuZoneBeginAllocCallstackSerial(TracyGpuZoneBeginCallstackData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_alloc_serial")]
    public static extern void TracyEmitGpuZoneBeginAllocSerial(TracyGpuZoneBeginData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_callstack")]
    public static extern void TracyEmitGpuZoneBeginCallstack(TracyGpuZoneBeginCallstackData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_callstack_serial")]
    public static extern void TracyEmitGpuZoneBeginCallstackSerial(TracyGpuZoneBeginCallstackData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_begin_serial")]
    public static extern void TracyEmitGpuZoneBeginSerial(TracyGpuZoneBeginData param);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_end")]
    public static extern void TracyEmitGpuZoneEnd(TracyGpuZoneEndData data);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_gpu_zone_end_serial")]
    public static extern void TracyEmitGpuZoneEndSerial(TracyGpuZoneEndData data);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_alloc")]
    public static extern void TracyEmitMemoryAlloc(void* ptr, ulong size, int secure);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_alloc_callstack")]
    public static extern void TracyEmitMemoryAllocCallstack(void* ptr, ulong size, int depth, int secure);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_alloc_callstack_named")]
    public static extern void TracyEmitMemoryAllocCallstackNamed(void* ptr, ulong size, int depth, int secure, CString name);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_alloc_named")]
    public static extern void TracyEmitMemoryAllocNamed(void* ptr, ulong size, int secure, CString name);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_free")]
    public static extern void TracyEmitMemoryFree(void* ptr, int secure);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_free_callstack")]
    public static extern void TracyEmitMemoryFreeCallstack(void* ptr, int depth, int secure);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_free_callstack_named")]
    public static extern void TracyEmitMemoryFreeCallstackNamed(void* ptr, int depth, int secure, CString name);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_memory_free_named")]
    public static extern void TracyEmitMemoryFreeNamed(void* ptr, int secure, CString name);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_message")]
    public static extern void TracyEmitMessage(CString txt, ulong size, int callstack);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_message_appinfo")]
    public static extern void TracyEmitMessageAppinfo(CString txt, ulong size);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_messageC")]
    public static extern void TracyEmitMessageC(CString txt, ulong size, uint color, int callstack);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_messageL")]
    public static extern void TracyEmitMessageL(CString txt, int callstack);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_messageLC")]
    public static extern void TracyEmitMessageLC(CString txt, uint color, int callstack);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_plot")]
    public static extern void TracyEmitPlot(CString name, Double val);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_plot_config")]
    public static extern void TracyEmitPlotConfig(CString name, int type, int step, int fill, uint color);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_plot_float")]
    public static extern void TracyEmitPlotFloat(CString name, float val);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_plot_int")]
    public static extern void TracyEmitPlotInt(CString name, long val);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_begin")]
    public static extern TracyCZoneCtx TracyEmitZoneBegin(TracySourceLocationData* srcloc, int active);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_begin_alloc")]
    public static extern TracyCZoneCtx TracyEmitZoneBeginAlloc(ulong srcloc, int active);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_begin_alloc_callstack")]
    public static extern TracyCZoneCtx TracyEmitZoneBeginAllocCallstack(ulong srcloc, int depth, int active);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_begin_callstack")]
    public static extern TracyCZoneCtx TracyEmitZoneBeginCallstack(TracySourceLocationData* srcloc, int depth, int active);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_color")]
    public static extern void TracyEmitZoneColor(TracyCZoneCtx ctx, uint color);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_end")]
    public static extern void TracyEmitZoneEnd(TracyCZoneCtx ctx);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_name")]
    public static extern void TracyEmitZoneName(TracyCZoneCtx ctx, CString txt, ulong size);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_text")]
    public static extern void TracyEmitZoneText(TracyCZoneCtx ctx, CString txt, ulong size);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_emit_zone_value")]
    public static extern void TracyEmitZoneValue(TracyCZoneCtx ctx, ulong value);

    [CNode(Kind = "Function")]
    [DllImport(LibraryName, EntryPoint = "___tracy_set_thread_name")]
    public static extern void TracySetThreadName(CString name);

    #endregion

    #region Types

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 8, Pack = 4)]
    public struct TracyCZoneContext
    {
        [FieldOffset(0)] // size = 4
        public uint Id;

        [FieldOffset(4)] // size = 4
        public int Active;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 24, Pack = 8)]
    public struct TracyGpuCalibrationData
    {
        [FieldOffset(0)] // size = 8
        public long GpuTime;

        [FieldOffset(8)] // size = 8
        public long CpuDelta;

        [FieldOffset(16)] // size = 1
        public byte Context;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 24, Pack = 8)]
    public struct TracyGpuContextNameData
    {
        [FieldOffset(0)] // size = 1
        public byte Context;

        [FieldOffset(8)] // size = 8
        public CString _Name;

        public string Name
        {
            get
            {
                return CString.ToString(_Name);
            }
            set
            {
                _Name = CString.FromString(value);
            }
        }

        [FieldOffset(16)] // size = 2
        public ushort Len;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
    public struct TracyGpuNewContextData
    {
        [FieldOffset(0)] // size = 8
        public long GpuTime;

        [FieldOffset(8)] // size = 4
        public float Period;

        [FieldOffset(12)] // size = 1
        public byte Context;

        [FieldOffset(13)] // size = 1
        public byte Flags;

        [FieldOffset(14)] // size = 1
        public byte Type;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
    public struct TracyGpuTimeData
    {
        [FieldOffset(0)] // size = 8
        public long GpuTime;

        [FieldOffset(8)] // size = 2
        public ushort QueryId;

        [FieldOffset(10)] // size = 1
        public byte Context;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
    public struct TracyGpuZoneBeginCallstackData
    {
        [FieldOffset(0)] // size = 8
        public ulong Srcloc;

        [FieldOffset(8)] // size = 4
        public int Depth;

        [FieldOffset(12)] // size = 2
        public ushort QueryId;

        [FieldOffset(14)] // size = 1
        public byte Context;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 16, Pack = 8)]
    public struct TracyGpuZoneBeginData
    {
        [FieldOffset(0)] // size = 8
        public ulong Srcloc;

        [FieldOffset(8)] // size = 2
        public ushort QueryId;

        [FieldOffset(10)] // size = 1
        public byte Context;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 4, Pack = 2)]
    public struct TracyGpuZoneEndData
    {
        [FieldOffset(0)] // size = 2
        public ushort QueryId;

        [FieldOffset(2)] // size = 1
        public byte Context;
    }

    [CNode(Kind = "Struct")]
    [StructLayout(LayoutKind.Explicit, Size = 32, Pack = 8)]
    public struct TracySourceLocationData
    {
        [FieldOffset(0)] // size = 8
        public CString _Name;

        public string Name
        {
            get
            {
                return CString.ToString(_Name);
            }
            set
            {
                _Name = CString.FromString(value);
            }
        }

        [FieldOffset(8)] // size = 8
        public CString _Function;

        public string Function
        {
            get
            {
                return CString.ToString(_Function);
            }
            set
            {
                _Function = CString.FromString(value);
            }
        }

        [FieldOffset(16)] // size = 8
        public CString _File;

        public string File
        {
            get
            {
                return CString.ToString(_File);
            }
            set
            {
                _File = CString.FromString(value);
            }
        }

        [FieldOffset(24)] // size = 4
        public uint Line;

        [FieldOffset(28)] // size = 4
        public uint Color;
    }

    [CNode(Kind = "Enum")]
    public enum TracyPlotFormatEnum : int
    {
        TracyPlotFormatNumber = 0,
        TracyPlotFormatMemory = 1,
        TracyPlotFormatPercentage = 2,
        TracyPlotFormatWatt = 3
    }

    [CNode(Kind = "TypeAlias")]
    [StructLayout(LayoutKind.Explicit, Size = 8, Pack = 4)]
    public struct TracyCZoneCtx
    {
        [FieldOffset(0)]
        public TracyCZoneContext Data;

        public static implicit operator TracyCZoneContext(TracyCZoneCtx data) => data.Data;
        public static implicit operator TracyCZoneCtx(TracyCZoneContext data) => new() { Data = data };
    }

    [CNode(Kind = "MacroObject")]
    public const int TracyHasCallstack = 1;

    #endregion
}
