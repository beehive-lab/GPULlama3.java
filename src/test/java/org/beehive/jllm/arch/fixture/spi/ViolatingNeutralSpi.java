package org.beehive.gpullama3.arch.fixture.spi;

import org.beehive.gpullama3.backend.tornado.TornadoVMMasterPlan;

/** Never referenced by production code. */
public class ViolatingNeutralSpi {

    /** A neutral contract that hands back an implementation type defeats the whole arrangement. */
    public TornadoVMMasterPlan leakTheImplementation() {
        return null;
    }
}
