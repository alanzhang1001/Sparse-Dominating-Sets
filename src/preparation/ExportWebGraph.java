import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.io.BufferedWriter;
import java.util.HashSet;
import java.util.Random;

/** Sequential decoding; one little-endian uint64 canonical edge per nonloop arc. */
public class ExportWebGraph {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) throw new IllegalArgumentException("BASENAME EDGES.bin RECEIPT.json");
        long begin = System.nanoTime();
        ImmutableGraph graph = BVGraph.loadSequential(args[0]);
        int n = graph.numNodes();
        long expected = graph.numArcs(), arcs = 0, loops = 0, written = 0;
        int nodes = 0;
        HashSet<Integer> auditNodes = new HashSet<>();
        Random random = new Random(20261013);
        auditNodes.add(0); auditNodes.add(n - 1);
        while (auditNodes.size() < Math.min(n, 128)) auditNodes.add(random.nextInt(n));
        ByteBuffer buffer = ByteBuffer.allocateDirect(8 * 1024 * 1024).order(ByteOrder.LITTLE_ENDIAN);
        try (FileOutputStream stream = new FileOutputStream(args[1]); FileChannel channel = stream.getChannel();
             BufferedWriter audit = Files.newBufferedWriter(Path.of(args[2] + ".directed-audit.tsv"))) {
            NodeIterator iterator = graph.nodeIterator();
            while (iterator.hasNext()) {
                int u = iterator.nextInt();
                if (u != nodes++) throw new IllegalStateException("nonconsecutive node IDs");
                int degree = iterator.outdegree();
                int[] successors = iterator.successorArray();
                boolean auditSource = auditNodes.contains(u);
                for (int j = 0; j < degree; j++) {
                    int v = successors[j];
                    if (v < 0 || v >= n) throw new IllegalStateException("invalid successor");
                    arcs++;
                    if (auditSource) audit.write(u + "\t" + v + "\n");
                    if (v == u) { loops++; continue; }
                    if (buffer.remaining() < 8) {
                        buffer.flip(); while (buffer.hasRemaining()) channel.write(buffer); buffer.clear();
                    }
                    buffer.putLong(((long)Math.min(u, v) << 32) | (Math.max(u, v) & 0xffffffffL));
                    written++;
                }
                if (nodes % 1000000 == 0) System.out.println("{\"decoded_nodes\":" + nodes + ",\"decoded_arcs\":" + arcs + "}");
            }
            buffer.flip(); while (buffer.hasRemaining()) channel.write(buffer);
        }
        if (nodes != n || arcs != expected) throw new IllegalStateException("decoded size differs from properties");
        long bytes = Files.size(Path.of(args[1]));
        if (bytes != written * 8) throw new IllegalStateException("truncated edge output");
        String result = "{\"nodes\":" + n + ",\"directed_arcs\":" + arcs + ",\"selfloops_removed\":" + loops
                + ",\"canonical_edge_records\":" + written + ",\"bytes\":" + bytes
                + ",\"seconds\":" + (System.nanoTime()-begin)/1e9 + "}";
        Files.writeString(Path.of(args[2]), result + "\n");
        System.out.println(result);
    }
}
