package org.beehive.gpullama3.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication(scanBasePackages = "org.beehive.gpullama3")
public class LLMApiApplication {

    public static void main(String[] args) {
        System.out.println("Starting TornadoVM LLM API Server...");
        System.out.println("Command line arguments: " + String.join(" ", args));

        // Let Options.parseOptions() handle validation - no duplication
        SpringApplication.run(LLMApiApplication.class, args);
    }
}