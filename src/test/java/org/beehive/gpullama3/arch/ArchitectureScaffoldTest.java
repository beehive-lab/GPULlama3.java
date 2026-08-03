package org.beehive.gpullama3.arch;

import com.tngtech.archunit.core.domain.JavaClasses;
import org.junit.Test;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * T1.1 scaffold. Proves the ArchUnit machinery is wired up and importing the production
 * tree, so that the rule tasks that follow (T1.2 rules 1/2/5/7/11, T1.3 rules 8a/16) only
 * have to add rules.
 *
 * <p>Deliberately asserts nothing about architecture yet — the rules land in T1.2/T1.3.
 */
public class ArchitectureScaffoldTest {

    @Test
    public void importsTheProductionTree() {
        JavaClasses classes = ProductionClasses.get();
        assertFalse("no production classes imported — check the module layout", classes.isEmpty());
        assertTrue("expected the production tree to be substantial", classes.size() > 100);
    }

    @Test
    public void excludesTestClasses() {
        boolean anyTestClass = ProductionClasses.get().stream()
                .anyMatch(c -> c.getName().endsWith("Test"));
        assertFalse("test classes must not be part of the rule input", anyTestClass);
    }
}
