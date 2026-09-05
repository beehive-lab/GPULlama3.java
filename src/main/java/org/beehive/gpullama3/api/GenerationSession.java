package org.beehive.gpullama3.api;

/**
 * One sequence: a conversation, a completion, one thread's worth of work.
 *
 * <p><b>Not thread-safe.</b> A session carries the sequence's position and its key/value state; two
 * threads generating into one session would interleave a single conversation. Run several sessions
 * instead — they share the model's weights and cost only their own context.
 *
 * <p>The session holds its key/value state as a lease rather than owning storage, so blocks can be
 * shared between sequences and can outlive an individual session.
 *
 * <p><b>Multi-turn is a session that stays open.</b> Each {@code generate} continues from the
 * position the last one reached; {@link #reset()} starts the conversation over without paying for a
 * new session.
 *
 * <p>{@link #close()} is idempotent. Closing the model with a session still open is an error — see
 * {@link LocalModel#close()}.
 */
public interface GenerationSession extends AutoCloseable {

    /**
     * Generates a completion, continuing this session's sequence.
     *
     * @throws IllegalStateException if the session or its model is closed
     */
    GenerationResult generate(GenerationRequest request);

    /** How much of the context this sequence has consumed, in tokens. */
    int position();

    /** Discards the sequence and starts from an empty context. The session stays usable. */
    void reset();

    /** Releases the sequence's state. Idempotent. */
    @Override
    void close();
}
