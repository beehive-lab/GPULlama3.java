package org.beehive.gpullama3.arch;

import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.Set;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/**
 * T1.2 acceptance: the rules must fail on a deliberately-violating fixture.
 *
 * <p>A rule that only ever passes proves nothing — it could be matching the wrong package or
 * silently returning empty. These tests point each rule at
 * {@code org.beehive.gpullama3.arch.fixture.model} and assert it reports the planted violations.
 */
public class DependencyRulesSelfTest {

    private static final String FIXTURE = "org.beehive.gpullama3.arch.fixture.model";
    private static final String FIXTURE_BACKEND = "org.beehive.gpullama3.arch.fixture.backend";

    private static JavaClasses fixture;

    @BeforeClass
    public static void importFixture() {
        fixture = new ClassFileImporter().importPackages(FIXTURE);
        assertFalse("fixture classes did not import", fixture.isEmpty());
    }

    @Test
    public void rule1_flagsTornadoVmOutsideTheBackend() {
        Set<String> v = ArchRules.rule1TornadoVmOutsideBackend(fixture, FIXTURE_BACKEND);
        assertTrue("Rule 1 did not flag the fixture: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule2_flagsModelDependingOnTornado() {
        Set<String> v = ArchRules.rule2ModelDependsOnTornado(fixture, FIXTURE, FIXTURE_BACKEND);
        assertTrue("Rule 2 did not flag the fixture: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule5_flagsMutableModelFields() {
        Set<String> v = ArchRules.rule5MutableModelFields(fixture,
                c -> c.getName().equals(FIXTURE + ".ViolatingModel"));
        assertTrue("Rule 5 did not flag the mutable field: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule7_flagsKvStorageReachableFromModel() {
        Set<String> v = ArchRules.rule7ModelReachesKvStorage(fixture, FIXTURE);
        assertTrue("Rule 7 did not flag the KV user: " + v, v.contains(FIXTURE + ".ViolatingKvUser"));
    }

    @Test
    public void rule11_flagsPlanTypesOutsideTheBackend() {
        Set<String> v = ArchRules.rule11PlanTypesOutsideBackend(fixture, FIXTURE_BACKEND);
        assertTrue("Rule 11 did not flag TaskGraph use: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rulesAreQuietWhenTheFixtureIsTreatedAsTheBackend() {
        // Same classes, but now inside the backend package: Rules 1 and 11 must go silent.
        assertTrue(ArchRules.rule1TornadoVmOutsideBackend(fixture, FIXTURE).isEmpty());
        assertTrue(ArchRules.rule11PlanTypesOutsideBackend(fixture, FIXTURE).isEmpty());
    }
}
