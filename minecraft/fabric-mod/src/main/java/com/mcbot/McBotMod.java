package com.mcbot;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class McBotMod implements ModInitializer {
    public static final String MOD_ID = "mcbot";

    private static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    private long observedServerTickCallbacks;

    @Override
    public void onInitialize() {
        LOGGER.info("mcbot initialized; no policy bridge is active");
        if (!"1".equals(System.getenv("MCBOT_TICK_TRACE"))) {
            return;
        }

        ServerTickEvents.START_SERVER_TICK.register(server ->
                LOGGER.info("server tick start observed_callback_count={}", observedServerTickCallbacks));
        ServerTickEvents.END_SERVER_TICK.register(server ->
                LOGGER.info("server tick end observed_callback_count={}", ++observedServerTickCallbacks));
    }
}
