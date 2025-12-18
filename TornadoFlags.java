// === Set to not get annoying warnings about annotation processing
//JAVAC_OPTIONS -proc:full

// === Deps for GraalVM compiler (needed for TornadoVM) ===
//DEPS org.graalvm.compiler:compiler:23.1.0

// === JVM mode and memory settings ===
//JAVA_OPTIONS -server
//JAVA_OPTIONS -XX:-UseCompressedOops
//JAVA_OPTIONS -XX:+UnlockExperimentalVMOptions
//JAVA_OPTIONS -XX:+EnableJVMCI
//JAVA_OPTIONS -XX:-UseCompressedClassPointers
//JAVA_OPTIONS -XX:+UseParallelGC

// === Native library path ===
//JAVA_OPTIONS -Djava.library.path=${env.TORNADOVM_HOME}/lib

// === Tornado runtime classes ===
//JAVA_OPTIONS -Dtornado.load.api.implementation=uk.ac.manchester.tornado.runtime.tasks.TornadoTaskGraph
//JAVA_OPTIONS -Dtornado.load.runtime.implementation=uk.ac.manchester.tornado.runtime.TornadoCoreRuntime
//JAVA_OPTIONS -Dtornado.load.tornado.implementation=uk.ac.manchester.tornado.runtime.common.Tornado
//JAVA_OPTIONS -Dtornado.load.annotation.implementation=uk.ac.manchester.tornado.annotation.ASMClassVisitor
//JAVA_OPTIONS -Dtornado.load.annotation.parallel=uk.ac.manchester.tornado.api.annotations.Parallel

// === Module system ===
//JAVA_OPTIONS --module-path ${env.TORNADOVM_HOME}/share/java/tornado
//JAVA_OPTIONS --upgrade-module-path ${env.TORNADOVM_HOME}/share/java/graalJars
//JAVA_OPTIONS --add-modules ALL-SYSTEM,tornado.runtime,tornado.annotation,tornado.drivers.common,tornado.drivers.opencl

// === Common exports ===
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.cfg=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.common=jdk.internal.vm.compiler
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.hotspot.meta=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.util=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.meta=tornado.runtime,tornado.annotation,tornado.drivers.common,jdk.internal.vm.compiler
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.code=tornado.runtime,tornado.drivers.common,jdk.internal.vm.compiler
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph.spi=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.gen=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodeinfo=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.calc=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.spi=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.api.runtime=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.code=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.target=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.debug=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.hotspot=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.java=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.asm=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.phases=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.graphbuilderconf=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.options=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.tiers=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.util=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.printer=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.runtime=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.runtime=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph.iterators=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.java=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.bytecode=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.spi=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.api.replacements=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.replacements=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.inlining=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.phases=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.type=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.extended=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.loop=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.inlining.info=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.inlining.policy=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.inlining.walker=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.loop.phases=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.debug=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.memory=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.util=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.virtual=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.constopt=tornado.runtime
//JAVA_OPTIONS --add-opens jdk.internal.vm.ci/jdk.vm.ci.hotspot=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.hotspot=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.gc=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.memory.address=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.replacements.nodes=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.word=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.util=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.framemap=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.alloc=tornado.runtime
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.memory=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph=tornado.runtime,tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph.iterators=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.java=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.extended=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.loop=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.calc=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.options=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.debug=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.util=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.virtual=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.loop.phases=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.util=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.tiers=tornado.drivers.common
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common=tornado.drivers.common

// === OpenCL-specific exports ===
//JAVA_OPTIONS --add-opens java.base/java.lang=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.common=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.amd64=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.hotspot.meta=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.replacements.classfile=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.alloc=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.util=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.cfg=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.framemap=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.meta=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.code=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph.spi=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.gen=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodeinfo=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.calc=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.spi=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.code=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.debug=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.hotspot=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.java=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.asm=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.phases=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.graphbuilderconf=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.options=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.tiers=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.util=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.printer=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.runtime=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.graph.iterators=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.java=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.bytecode=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.spi=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.api.replacements=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.replacements=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.inlining=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.phases=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.type=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.extended=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.loop=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.loop.phases=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.debug=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.memory=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.util=tornado.drivers.opencl
//JAVA_OPTIONS --add-opens jdk.internal.vm.ci/jdk.vm.ci.hotspot=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.ci/jdk.vm.ci.hotspot=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.asm=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.cfg=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.schedule=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.virtual.phases.ea=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.lir.ssa=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.calc=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.gen=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.match=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.memory.address=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.nodes.type=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.graph=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.phases.common.util=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.word=tornado.drivers.opencl
//JAVA_OPTIONS --add-exports jdk.internal.vm.compiler/org.graalvm.compiler.core.common.memory=tornado.drivers.opencl

package org.beehive.gpullama3.cli;

/**
 * TornadoFlags - JBang configuration file for TornadoVM runtime setup
 *
 * This file contains all the JVM options and module exports needed to run
 * TornadoVM with JBang. It's referenced from LlamaTornadoCli.java using:
 * //SOURCES TornadoFlags.java
 *
 * This pattern keeps the main CLI file clean while ensuring all necessary
 * TornadoVM runtime configuration is properly set up.
 */
public class TornadoFlags {
    // This class is intentionally empty - all configuration is in JBang directives above
}
