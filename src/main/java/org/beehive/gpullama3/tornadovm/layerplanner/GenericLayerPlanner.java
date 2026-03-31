package org.beehive.gpullama3.tornadovm.layerplanner;

import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;

import java.util.List;

public interface GenericLayerPlanner {

    List<ImmutableTaskGraph> getImmutableTaskGraphs();

    GridScheduler getGridScheduler();

}
