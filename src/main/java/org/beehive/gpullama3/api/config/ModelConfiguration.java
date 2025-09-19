package org.beehive.gpullama3.api.config;

import org.beehive.gpullama3.api.service.LLMService;
import org.beehive.gpullama3.model.Model;
import org.beehive.gpullama3.Options;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class ModelConfiguration {

    /**
     * Expose Model as a Spring bean using the initialized service
     */
    @Bean
    public Model model(LLMService llmService) {
        return llmService.getModel();
    }

    /**
     * Expose Options as a Spring bean using the initialized service
     */
    @Bean
    public Options options(LLMService llmService) {
        return llmService.getOptions();
    }
}