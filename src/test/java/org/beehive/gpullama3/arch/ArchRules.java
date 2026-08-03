package org.beehive.gpullama3.arch;

import com.tngtech.archunit.core.domain.JavaClass;
import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.domain.JavaModifier;

import java.util.Set;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * The dependency rules of {@code docs/architecture/dependency-rules.md}, each expressed as a
 * function from imported classes to the set of violating class names.
 *
 * <p>Returning the violators rather than asserting directly is deliberate: it lets the same
 * rule run against the production tree (compared to an allowlist) and against a deliberately
 * violating fixture (to prove the rule actually bites). A rule that is never seen to fail is
 * not a guardrail.
 *
 * <p><b>Package mapping.</b> The rule documents are written against the target package names
 * from {@code target-architecture.md}, which do not exist yet. Today's equivalents:
 * {@code ..backend.tornado..} is {@code org.beehive.gpullama3.tornadovm..}, and there is not
 * yet a {@code generation}/{@code api}/{@code integration} split. The package prefixes are
 * therefore parameters, so the rules survive the M12 package move without being rewritten.
 */
public final class ArchRules {

    /** TornadoVM's own packages — the thing Rules 1 and 2 keep out of the upper layers. */
    public static final String TORNADO_VM = "uk.ac.manchester.tornado";

    /** Today's stand-in for {@code ..backend.tornado..}. */
    public static final String TORNADO_BACKEND = "org.beehive.gpullama3.tornadovm";

    public static final String MODEL = "org.beehive.gpullama3.model";

    /** Rule 11's type-specific list — the plan types most likely to leak out of the backend. */
    public static final Set<String> PLAN_TYPES = Set.of(
            "uk.ac.manchester.tornado.api.TaskGraph",
            "uk.ac.manchester.tornado.api.ImmutableTaskGraph",
            "uk.ac.manchester.tornado.api.TornadoExecutionPlan",
            "uk.ac.manchester.tornado.api.GridScheduler");

    private ArchRules() {
    }

    /** Rule 1 — TornadoVM stays in the Tornado backend. */
    public static Set<String> rule1TornadoVmOutsideBackend(JavaClasses classes, String backendPrefix) {
        return violators(classes, c -> !inPackage(c, backendPrefix) && dependsOnPackage(c, TORNADO_VM));
    }

    /**
     * Rule 2 — model architecture packages do not import TornadoVM. Broader than Rule 1 inside
     * {@code model}: depending on the backend package counts too, because
     * {@code TornadoVMMasterPlan} is Tornado-specific without being a {@code uk.ac.manchester}
     * import.
     */
    public static Set<String> rule2ModelDependsOnTornado(JavaClasses classes, String modelPrefix, String backendPrefix) {
        return violators(classes, c -> inPackage(c, modelPrefix)
                && (dependsOnPackage(c, TORNADO_VM) || dependsOnPackage(c, backendPrefix)));
    }

    /**
     * Rule 5 — models own immutable configuration and weights. Scoped to loaded-model types;
     * loaders and builders are explicitly out of scope per the rule text.
     */
    public static Set<String> rule5MutableModelFields(JavaClasses classes, Predicate<JavaClass> loadedModelTypes) {
        return violators(classes, c -> loadedModelTypes.test(c) && hasNonFinalField(c));
    }

    /** Rule 7 — KV storage is never reachable from a model or a program. */
    public static Set<String> rule7ModelReachesKvStorage(JavaClasses classes, String... upperPrefixes) {
        return violators(classes, c -> {
            for (String p : upperPrefixes) {
                if (inPackage(c, p)) {
                    return c.getDirectDependenciesFromSelf().stream().anyMatch(d -> {
                        String n = d.getTargetClass().getSimpleName();
                        return n.contains("KvCache") || n.contains("BlockPool");
                    });
                }
            }
            return false;
        });
    }

    /**
     * Today's stand-in for {@code ..generation..}, {@code ..api..} and {@code ..integration..}:
     * the CLI entry point, the CLI options record and the HTTP server. There is no
     * {@code integration.cli} package yet, so the CLI is identified by type rather than package.
     */
    public static final Set<String> CLI_TYPES = Set.of(
            "org.beehive.gpullama3.LlamaApp",
            "org.beehive.gpullama3.Options");

    public static final String SERVER = "org.beehive.gpullama3.server";

    /**
     * Rule 8a — generation policy is separate from forward execution. Lower layers must not
     * reach the CLI, the options record or the server integration. The integrations themselves
     * are excluded: depending on generation policy is their job.
     */
    public static Set<String> rule8aLowerLayersDependOnGenerationPolicy(JavaClasses classes) {
        return violators(classes, c -> !isIntegration(c)
                && c.getDirectDependenciesFromSelf().stream().anyMatch(d -> {
                    String n = d.getTargetClass().getName();
                    return CLI_TYPES.contains(n) || n.startsWith(SERVER + ".");
                }));
    }

    /**
     * Rule 16 — no console I/O outside the CLI integration. Library code that prints cannot be
     * silenced or routed by an embedder.
     */
    public static Set<String> rule16ConsoleIoOutsideCli(JavaClasses classes) {
        return violators(classes, c -> !isIntegration(c) && printsToConsole(c));
    }

    private static boolean isIntegration(JavaClass c) {
        String outer = c.getName().split("\\$")[0];
        return CLI_TYPES.contains(outer) || inPackage(c, SERVER);
    }

    private static boolean printsToConsole(JavaClass c) {
        return c.getMethodCallsFromSelf().stream().anyMatch(call -> {
            String m = call.getName();
            return call.getTargetOwner().getName().equals("java.io.PrintStream")
                    && (m.equals("println") || m.equals("print") || m.equals("printf"));
        });
    }

    /** Rule 11 — TaskGraph / ImmutableTaskGraph / TornadoExecutionPlan / GridScheduler stay in the backend. */
    public static Set<String> rule11PlanTypesOutsideBackend(JavaClasses classes, String backendPrefix) {
        return violators(classes, c -> !inPackage(c, backendPrefix)
                && c.getDirectDependenciesFromSelf().stream()
                        .anyMatch(d -> PLAN_TYPES.contains(d.getTargetClass().getName())));
    }

    // helpers

    private static Set<String> violators(JavaClasses classes, Predicate<JavaClass> violates) {
        Set<String> out = new TreeSet<>();
        for (JavaClass c : classes) {
            if (violates.test(c)) {
                out.add(c.getName());
            }
        }
        return out;
    }

    private static boolean inPackage(JavaClass c, String prefix) {
        String p = c.getPackageName();
        return p.equals(prefix) || p.startsWith(prefix + ".");
    }

    private static boolean dependsOnPackage(JavaClass c, String prefix) {
        return c.getDirectDependenciesFromSelf().stream()
                .anyMatch(d -> {
                    String p = d.getTargetClass().getPackageName();
                    return p.equals(prefix) || p.startsWith(prefix + ".");
                });
    }

    private static boolean hasNonFinalField(JavaClass c) {
        return c.getFields().stream().anyMatch(f -> !f.getModifiers().contains(JavaModifier.FINAL));
    }
}
