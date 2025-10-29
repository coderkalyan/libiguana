const std = @import("std");

const testing = std.testing;
const Allocator = std.mem.Allocator;

/// Implementation of an interning string pool that allocates strings
/// (u8 buffers) into a single underlying buffer and uniques them. This
/// provides:
/// * Significantly better allocation performance, as all strings are backed
/// by a single allocation.
/// * Individual lifetime management is replaced by a single lifetime for the
/// entire string pool.
/// * Lower internal fragmentation and better cache locality than normal
/// allocations. Each string has only 1 byte of overhead.
/// * String interning, which is fairly cheap and reduces memory usage
/// significantly in some workloads.
/// * Extremely cheap equality comparison through integer handles.
///
/// Initialize using `.empty`.
///
/// Important usage notes:
/// * The pool can store at most 4GiB of string data, to minimize handle size.
/// * The data structure is *unmanaged* so users pass allocators to allocating functions.
/// * The implementation is not currently thread safe.
pub const StringPool = struct {
    /// Backing store for interned strings.
    bytes: std.ArrayListUnmanaged(u8),
    /// Probing table for interning strings in the `bytes` backing store.
    table: std.HashMapUnmanaged(Ref, void, RefContext, std.hash_map.default_max_load_percentage),

    pub const empty: StringPool = .{
        .bytes = .empty,
        .table = .empty,
    };

    /// Opaque 32 bit handle to interned strings.
    pub const Ref = enum(u32) { _ };

    /// This context should be treated as ephemeral as the underlying bytes slice is invalidated
    /// after the next insert to the underlying ArrayList.
    const RefContext = struct {
        bytes: []const u8,

        pub fn eql(self: RefContext, a: Ref, b: Ref) bool {
            _ = self;
            return a == b;
        }

        pub fn hash(self: RefContext, index: Ref) u64 {
            const x: u32 = @intFromEnum(index);
            const str = std.mem.span(@as([*:0]const u8, @ptrCast(self.bytes.ptr)) + x);
            return std.hash_map.hashString(str);
        }
    };

    /// This adapter should be treated as ephemeral as the underlying bytes slice is invalidated
    /// after the next insert to the underlying ArrayList.
    const SliceAdapter = struct {
        bytes: []const u8,

        pub fn eql(self: SliceAdapter, a_str: []const u8, b: Ref) bool {
            const offset = @intFromEnum(b);
            const b_str = std.mem.span(@as([*:0]const u8, @ptrCast(self.bytes.ptr)) + offset);
            return std.mem.eql(u8, a_str, b_str);
        }

        pub fn hash(self: SliceAdapter, str: []const u8) u64 {
            _ = self;
            return std.hash_map.hashString(str);
        }
    };

    /// Deinitialize the string pool, freeing all underlying memory.
    pub fn deinit(pool: *StringPool, gpa: Allocator) void {
        pool.bytes.deinit(gpa);
        pool.table.deinit(gpa);
    }

    /// Ensure there is sufficient capacity in the string pool to intern a
    /// string of `len` bytes.
    pub fn ensureUnusedCapacity(pool: *StringPool, gpa: Allocator, len: usize) !void {
        // Ensure the bytes table does not overflow a u32 index (as usize may be u64).
        // One additional byte is required for the internal null terminator.
        std.debug.assert(pool.bytes.items.len < std.math.maxInt(u32) - len - 1);

        const context: RefContext = .{ .bytes = pool.bytes.items };
        try pool.bytes.ensureUnusedCapacity(gpa, len + 1);
        try pool.table.ensureUnusedCapacityContext(gpa, 1, context);
    }

    /// Intern a single string into the pool by inserting if needed, and return
    /// a `Ref` opaque handle to the string. Assumes there is sufficient
    /// capacity in the pool to insert the provided string.
    pub fn putAssumeCapacity(pool: *StringPool, str: []const u8) Ref {
        const adapter: SliceAdapter = .{ .bytes = pool.bytes.items };
        if (pool.table.getEntryAdapted(str, adapter)) |entry| {
            return entry.key_ptr.*;
        }

        const context: RefContext = .{ .bytes = pool.bytes.items };
        const index: u32 = @intCast(pool.bytes.items.len);
        const ref: Ref = @enumFromInt(index);
        pool.bytes.appendSliceAssumeCapacity(str);
        pool.bytes.appendAssumeCapacity('\x00');
        pool.table.putAssumeCapacityContext(ref, {}, context);
        return ref;
    }

    /// Intern a single string into the pool by inserting if needed, and return
    /// a `Ref` opaque handle to the string. This function allocates using the
    /// provided allocator.
    pub fn put(pool: *StringPool, gpa: Allocator, str: []const u8) !Ref {
        try pool.ensureUnusedCapacity(gpa, str.len);
        // No errors beyond this point.
        errdefer comptime unreachable;

        return pool.putAssumeCapacity(str);
    }

    /// Checks if the string pool contains the specified string buffer, without
    /// inserting or otherwise modifying the buffer.
    pub fn contains(pool: *const StringPool, str: []const u8) bool {
        const adapter: SliceAdapter = .{ .bytes = pool.bytes.items };
        return pool.table.containsAdapted(str, adapter);
    }

    /// Retrieves a slice to the underlying string buffer associated with a
    /// `Ref` handle that was previously interned into this pool.
    ///
    /// The returned slice is *not* owned by the caller. It's lifetime ends
    /// at the next call to `put` with any string (but lives across the next
    /// `putAssumeCapacity`)
    pub fn get(pool: *const StringPool, ref: Ref) []const u8 {
        return pool.slice().get(ref);
    }

    /// Retrieve a temporary read-only view into the StringPool. This does *not*
    /// copy or transfer ownership, but the lifetime of the returned `Slice` ends
    /// at the next call to `put` with any string.
    pub fn slice(pool: *const StringPool) Slice {
        return .{
            .bytes = pool.bytes.items,
        };
    }

    /// Moves the data in this pool into a newly allocated read-only StringPool.
    /// Ownership is transfered to the resulting slice and its data must be freed
    /// with `deinit`. It is not necessary to call `deinit` on the pool after this
    /// function, but is allowed.
    pub fn toOwnedSlice(pool: *StringPool, gpa: Allocator) !Slice {
        const bytes = try pool.bytes.toOwnedSlice(gpa);
        // No errors beyond this point.
        errdefer comptime unreachable;

        return .{
            .bytes = bytes,
        };
    }

    /// Read only reference to the StringPool. This is useful when you no longer
    /// plan to intern new objects into the pool, but need to be able to fetch
    /// existing strings by reference and want a read-only or lighter weight struct.
    pub const Slice = struct {
        /// Slice to the underlying string buffer memory.
        bytes: []const u8,

        /// Retrieves a slice to the underlying string buffer associated with a
        /// `Ref` handle that was previously interned into this pool.
        ///
        /// The returned slice is *not* owned by the caller. It's lifetime covers
        /// the lifetime of the parent Slice.
        pub fn get(pool: Slice, ref: Ref) []const u8 {
            const offset = @intFromEnum(ref);
            return std.mem.span(@as([*:0]const u8, @ptrCast(pool.bytes.ptr)) + offset);
        }

        /// Deinitializes the slice, freeing all underlying memory.
        pub fn deinit(pool: Slice, gpa: Allocator) void {
            gpa.free(pool.bytes);
        }
    };
};

test "insert" {
    const gpa = std.testing.allocator;
    var pool: StringPool = .empty;
    defer pool.deinit(gpa);

    const a_str = "apple";
    const b_str = "banana";
    const c_str = "cherry";
    // This testcase has complete overlap with `a_str` but should not deduplicate.
    const d_str = "apples";

    // Insert some strings into the pool.
    const a_ref = try pool.put(gpa, a_str);
    const b_ref = try pool.put(gpa, b_str);
    const c_ref = try pool.put(gpa, c_str);
    const d_ref = try pool.put(gpa, d_str);

    // All inserted strings are unique so the refs should be different.
    // Refs are opaque so we don't actually care *what* they are.
    inline for (&.{ b_ref, c_ref, d_ref }) |ref| try testing.expect(a_ref != ref);

    // All inserted strings should be in the pool.
    inline for (&.{ a_str, b_str, c_str, d_str }) |str| try testing.expect(pool.contains(str));

    // Fetch the strings and test the roundtrip.
    try testing.expectEqualSlices(u8, a_str, pool.get(a_ref));
    try testing.expectEqualSlices(u8, b_str, pool.get(b_ref));
    try testing.expectEqualSlices(u8, c_str, pool.get(c_ref));
    try testing.expectEqualSlices(u8, d_str, pool.get(d_ref));

    // Insert the same strings again and make sure we get the same value.
    try testing.expectEqual(a_ref, try pool.put(gpa, a_str));
    try testing.expectEqual(b_ref, try pool.put(gpa, b_str));
    try testing.expectEqual(c_ref, try pool.put(gpa, c_str));
    try testing.expectEqual(d_ref, try pool.put(gpa, d_str));
}

test "slice" {
    const gpa = std.testing.allocator;
    var pool: StringPool = .empty;
    defer pool.deinit(gpa);

    const a_str = "apple";
    const b_str = "banana";
    const c_str = "cherry";
    // This testcase has complete overlap with `a_str` but should not deduplicate.
    const d_str = "apples";

    // Insert some strings into the pool.
    const a_ref = try pool.put(gpa, a_str);
    const b_ref = try pool.put(gpa, b_str);
    const c_ref = try pool.put(gpa, c_str);
    const d_ref = try pool.put(gpa, d_str);

    // All inserted strings are unique so the refs should be different.
    // Refs are opaque so we don't actually care *what* they are.
    inline for (&.{ b_ref, c_ref, d_ref }) |ref| try testing.expect(a_ref != ref);

    // Test a "borrowed" read-only view.
    {
        const slice = pool.slice();
        // Fetch the strings and test the roundtrip.
        try testing.expectEqualSlices(u8, a_str, slice.get(a_ref));
        try testing.expectEqualSlices(u8, b_str, slice.get(b_ref));
        try testing.expectEqualSlices(u8, c_str, slice.get(c_ref));
        try testing.expectEqualSlices(u8, d_str, slice.get(d_ref));
    }

    // Test an owned read-only copy.
    {
        const slice = try pool.toOwnedSlice(gpa);
        defer slice.deinit(gpa);

        // Fetch the strings and test the roundtrip.
        try testing.expectEqualSlices(u8, a_str, slice.get(a_ref));
        try testing.expectEqualSlices(u8, b_str, slice.get(b_ref));
        try testing.expectEqualSlices(u8, c_str, slice.get(c_ref));
        try testing.expectEqualSlices(u8, d_str, slice.get(d_ref));
    }
}

test "oom" {
    const gpa = std.testing.failing_allocator;
    var pool: StringPool = .empty;
    defer pool.deinit(gpa);

    // Preconditions - expect the pool to be empty.
    try testing.expectEqual(0, pool.bytes.items.len);
    try testing.expectEqual(0, pool.table.size);

    // Inserting should fail with a failing allocator.
    const a_str = "apple";
    try testing.expectError(error.OutOfMemory, pool.put(gpa, a_str));

    // The underlying state should not have been modified - no partial inserts.
    try testing.expectEqual(0, pool.bytes.items.len);
    try testing.expectEqual(0, pool.table.size);
    try testing.expect(!pool.contains(a_str));
}
