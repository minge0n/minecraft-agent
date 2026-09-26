package com.mcbot;

record WorldRequest(long seed, Preset preset) {
    enum Preset { FLAT, NORMAL }
}
