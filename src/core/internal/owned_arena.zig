const std = @import("std");

/// A movable owner of a heap-stable arena. Do not copy an owner and free both.
pub const OwnedArena = struct {
    parent: std.mem.Allocator,
    arena: ?*std.heap.ArenaAllocator,

    pub fn init(parent: std.mem.Allocator) !OwnedArena {
        const arena = try parent.create(std.heap.ArenaAllocator);
        arena.* = .init(parent);
        return .{ .parent = parent, .arena = arena };
    }

    pub fn allocator(self: OwnedArena) std.mem.Allocator {
        return self.arena.?.allocator();
    }

    pub fn deinit(self: *OwnedArena) void {
        if (self.arena) |arena| {
            arena.deinit();
            self.parent.destroy(arena);
            self.arena = null;
        }
    }
};
