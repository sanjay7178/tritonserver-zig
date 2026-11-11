const std = @import("std");

/// Intrusive many-producer/single-consumer queue. Nodes have `queue_next: ?*T`.
/// Producers only touch their own unpublished node and the atomic head. The
/// consumer detaches a complete list and reverses it into publication order.
/// Unlike a linked-tail queue, a paused producer cannot leave a missing link.
pub fn MpscQueue(comptime T: type) type {
    return struct {
        head: std.atomic.Value(?*T) = .init(null),

        pub fn push(self: *@This(), node: *T) void {
            var previous = self.head.load(.acquire);
            while (true) {
                node.queue_next = previous;
                if (self.head.cmpxchgWeak(previous, node, .acq_rel, .acquire)) |actual| {
                    previous = actual;
                } else break;
            }
        }

        /// Only the consumer calls take. Detached nodes belong to the consumer
        /// and may be reclaimed: producers never dereference a previous head.
        pub fn take(self: *@This()) ?*T {
            var list = self.head.swap(null, .acq_rel);
            var ordered: ?*T = null;
            while (list) |node| {
                const next = node.queue_next;
                node.queue_next = ordered;
                ordered = node;
                list = next;
            }
            return ordered;
        }
    };
}

test "MPSC detaches FIFO lists and supports reuse after draining" {
    const Node = struct { value: usize, queue_next: ?*@This() = null };
    var queue: MpscQueue(Node) = .{};
    var nodes = [_]Node{ .{ .value = 1 }, .{ .value = 2 }, .{ .value = 3 } };
    for (&nodes) |*node| queue.push(node);
    var list = queue.take();
    for (1..4) |expected| {
        const node = list.?;
        try std.testing.expectEqual(expected, node.value);
        list = node.queue_next;
    }
    try std.testing.expect(list == null and queue.take() == null);
    queue.push(&nodes[0]);
    try std.testing.expect(queue.take() == &nodes[0]);
}

test "MPSC concurrent producers publish every node once in producer order" {
    const Node = struct { producer: usize, sequence: usize, queue_next: ?*@This() = null };
    const Queue = MpscQueue(Node);
    const Producer = struct {
        fn run(queue: *Queue, nodes: []Node) void {
            for (nodes) |*node| queue.push(node);
        }
    };
    const producer_count = 8;
    const per_producer = 1000;
    const nodes = try std.testing.allocator.alloc(Node, producer_count * per_producer);
    defer std.testing.allocator.free(nodes);
    var queue: Queue = .{};
    var threads: [producer_count]std.Thread = undefined;
    var started: usize = 0;
    defer for (threads[0..started]) |thread| thread.join();
    for (0..producer_count) |p| {
        const items = nodes[p * per_producer ..][0..per_producer];
        for (items, 0..) |*node, i| node.* = .{ .producer = p, .sequence = i };
        threads[started] = try std.Thread.spawn(.{}, Producer.run, .{ &queue, items });
        started += 1;
    }
    var observed: [producer_count]usize = @splat(0);
    var total: usize = 0;
    while (total < nodes.len) {
        var list = queue.take();
        while (list) |node| {
            try std.testing.expectEqual(observed[node.producer], node.sequence);
            observed[node.producer] += 1;
            total += 1;
            list = node.queue_next;
        }
        std.atomic.spinLoopHint();
    }
    for (threads[0..started]) |thread| thread.join();
    started = 0;
    try std.testing.expect(queue.take() == null);
}
