/**
 * Runnable examples of the public API.
 *
 * <p>Each class here is a {@code main} that does one thing and prints what it did. They are the
 * worked answer to "how do I use this from my own code": they import only {@code api} types, they
 * never touch a backend, a task graph or a GGUF type, and they are the same shape an embedder's
 * code should be.
 *
 * <p>Run one with the launcher's classpath, passing a model file:
 *
 * <pre>
 *   java -cp target/gpu-llama3-1.0.0-jdk21.jar \
 *        org.beehive.gpullama3.examples.HelloGeneration model.gguf
 * </pre>
 *
 * <p>To run on an accelerator, launch through {@code llama-tornado}, which sets the JVM flags and
 * {@code -Duse.tornadovm=true} that {@link org.beehive.gpullama3.api.ModelOptions} reads.
 */
package org.beehive.gpullama3.examples;
