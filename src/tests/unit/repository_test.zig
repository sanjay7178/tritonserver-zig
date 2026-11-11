const std = @import("std");
const t = @import("../../root.zig");
const testing = std.testing;

test "repository scan sorts models and positive numeric versions and ignores files" {
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    for ([_][]const u8{ "z/10", "z/2", "z/0", "z/-1", "z/not-a-version", "a/1", ".hidden/1" }) |path|
        try temp.dir.createDirPath(testing.io, path);
    try temp.dir.writeFile(testing.io, .{ .sub_path = "notes.txt", .data = "not a model" });
    try temp.dir.writeFile(testing.io, .{ .sub_path = "z/3", .data = "not a version directory" });
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var repo = try t.Repository.scan(testing.allocator, testing.io, path);
    defer repo.deinit();
    try testing.expectEqual(@as(usize, 2), repo.models.len);
    try testing.expectEqualStrings("a", repo.models[0].name);
    try testing.expectEqualSlices(i64, &.{ 2, 10 }, repo.models[1].versions);
    repo.deinit();
}

test "repository scan handles empty, missing and invalid paths" {
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    const path = try temp.dir.realPathFileAlloc(testing.io, ".", testing.allocator);
    defer testing.allocator.free(path);
    var repo = try t.Repository.scan(testing.allocator, testing.io, path);
    defer repo.deinit();
    try testing.expectEqual(@as(usize, 0), repo.models.len);
    const missing = try std.fs.path.join(testing.allocator, &.{ path, "missing" });
    defer testing.allocator.free(missing);
    try testing.expectError(error.FileNotFound, t.Repository.scan(testing.allocator, testing.io, missing));
    try testing.expectError(error.InvalidPath, t.Repository.scan(testing.allocator, testing.io, "a\x00b"));
}
