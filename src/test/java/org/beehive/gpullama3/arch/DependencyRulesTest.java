package org.beehive.gpullama3.arch;

import com.tngtech.archunit.core.domain.JavaClass;
import com.tngtech.archunit.core.domain.JavaClasses;
import org.junit.Test;

import java.util.Set;
import java.util.TreeSet;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

/**
 * T1.2 — Rules 1, 2, 5, 7 and 11 enforced against the production tree.
 *
 * <p>Each rule is compared to its allowlist in <b>both</b> directions. A class that violates
 * without being listed fails the build, which is the point of the rule; a listed class that no
 * longer violates <i>also</i> fails, because stale entries hide progress and would let an
 * allowlist quietly stop shrinking (policy item 5).
 */
public class DependencyRulesTest {

    private static JavaClasses classes() {
        return ProductionClasses.get();
    }

    @Test
    public void rule1_tornadoVmStaysInTheTornadoBackend() {
        assertMatchesAllowlist("Rule 1 (TornadoVM outside the Tornado backend)",
                ArchRules.rule1TornadoVmOutsideBackend(classes(), ArchRules.TORNADO_BACKEND),
                Allowlists.RULE_1);
    }

    @Test
    public void rule2_modelPackagesDoNotImportTornado() {
        assertMatchesAllowlist("Rule 2 (model depends on TornadoVM or the Tornado backend)",
                ArchRules.rule2ModelDependsOnTornado(classes(), ArchRules.MODEL, ArchRules.TORNADO_BACKEND),
                Allowlists.RULE_2);
    }

    @Test
    public void rule5_loadedModelsHaveOnlyFinalFields() {
        assertMatchesAllowlist("Rule 5 (mutable fields on loaded-model types)",
                ArchRules.rule5MutableModelFields(classes(), DependencyRulesTest::isLoadedModelType),
                Allowlists.RULE_5);
    }

    @Test
    public void rule7_kvStorageIsNotReachableFromModels() {
        Set<String> violations = ArchRules.rule7ModelReachesKvStorage(classes(), ArchRules.MODEL);
        assertTrue("Rule 7 passes today and must stay passing; new violations: " + violations,
                violations.isEmpty());
    }

    @Test
    public void rule11_planTypesStayInTheTornadoBackend() {
        Set<String> violations = ArchRules.rule11PlanTypesOutsideBackend(classes(), ArchRules.TORNADO_BACKEND);
        assertTrue("Rule 11 passes today and must stay passing; new violations: " + violations,
                violations.isEmpty());
    }

    @Test
    public void allowlistEntriesAreFullyQualifiedNames() {
        for (Set<String> list : Set.of(Allowlists.RULE_1, Allowlists.RULE_2, Allowlists.RULE_5)) {
            for (String entry : list) {
                assertTrue("wildcards are banned in allowlists: " + entry,
                        !entry.contains("*") && !entry.contains(".."));
                assertTrue("allowlist entries must be fully qualified: " + entry,
                        entry.startsWith(ProductionClasses.ROOT_PACKAGE + "."));
            }
        }
    }

    /** Loaded-model types only — loaders and builders are out of Rule 5's scope by its own text. */
    private static boolean isLoadedModelType(JavaClass c) {
        return !c.isInterface()
                && c.getAllRawSuperclasses().stream().anyMatch(s -> s.getName().equals("org.beehive.gpullama3.model.AbstractModel"))
                || c.getName().equals("org.beehive.gpullama3.model.AbstractModel");
    }

    private static void assertMatchesAllowlist(String rule, Set<String> actual, Set<String> allowed) {
        Set<String> unlisted = new TreeSet<>(actual);
        unlisted.removeAll(allowed);
        Set<String> stale = new TreeSet<>(allowed);
        stale.removeAll(actual);

        if (unlisted.isEmpty() && stale.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(rule).append(" failed.\n");
        if (!unlisted.isEmpty()) {
            sb.append("  NEW violations (fix the code, or record a maintainer decision to add them):\n");
            unlisted.forEach(v -> sb.append("    ").append(v).append('\n'));
        }
        if (!stale.isEmpty()) {
            sb.append("  STALE allowlist entries (these no longer violate — delete them):\n");
            stale.forEach(v -> sb.append("    ").append(v).append('\n'));
        }
        fail(sb.toString());
    }
}
