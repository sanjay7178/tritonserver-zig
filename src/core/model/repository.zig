const std = @import("std");
const OwnedArena = @import("../internal/owned_arena.zig").OwnedArena;

pub const RepositoryModel = struct {
    name: []const u8,
    versions: []const i64,
};

/// Owned filesystem inventory, independent of Triton's runtime model index.
/// Symlinks are not followed. Move this value; do not duplicate ownership.
pub const Repository = struct {
    storage: OwnedArena,
    path: []const u8,
    models: []const RepositoryModel,

    pub fn scan(parent: std.mem.Allocator, io: std.Io, path: []const u8) !Repository {
        if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidPath;
        var storage = try OwnedArena.init(parent);
        errdefer storage.deinit();
        const alloc = storage.allocator();
        var root = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
        defer root.close(io);
        const resolved = try root.realPathFileAlloc(io, ".", alloc);
        var models: std.ArrayList(RepositoryModel) = .empty;
        var it = root.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .directory or std.mem.startsWith(u8, entry.name, ".")) continue;
            const name = try alloc.dupe(u8, entry.name);
            var dir = try root.openDir(io, name, .{ .iterate = true });
            defer dir.close(io);
            var versions: std.ArrayList(i64) = .empty;
            var children = dir.iterate();
            while (try children.next(io)) |child| {
                if (child.kind != .directory or child.name.len == 0) continue;
                var digits = true;
                for (child.name) |ch| if (ch < '0' or ch > '9') {
                    digits = false;
                    break;
                };
                if (!digits) continue;
                const version = std.fmt.parseInt(i64, child.name, 10) catch continue;
                if (version > 0) try versions.append(alloc, version);
            }
            std.mem.sort(i64, versions.items, {}, std.sort.asc(i64));
            try models.append(alloc, .{ .name = name, .versions = try versions.toOwnedSlice(alloc) });
        }
        std.mem.sort(RepositoryModel, models.items, {}, struct {
            fn less(_: void, a: RepositoryModel, b: RepositoryModel) bool {
                return std.mem.lessThan(u8, a.name, b.name);
            }
        }.less);
        return .{ .storage = storage, .path = resolved, .models = try models.toOwnedSlice(alloc) };
    }

    pub fn deinit(self: *Repository) void {
        self.storage.deinit();
        self.path = "";
        self.models = &.{};
    }
};
