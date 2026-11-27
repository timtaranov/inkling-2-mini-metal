# Inkling-2-Mini inference server for Apple M5 Max
CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc -Ibuild -Ivendor
OBJCFLAGS := $(CXXFLAGS) -fobjc-arc
LDFLAGS  := -framework Metal -framework Foundation -framework Accelerate
BUILD    := build

ENGINE_OBJS := $(BUILD)/engine.o $(BUILD)/gguf.o
SERVER_OBJS := $(BUILD)/main.o $(BUILD)/server.o $(BUILD)/chat.o $(BUILD)/sampler.o $(BUILD)/tokenizer.o $(BUILD)/httplib.o

all: inkling-server $(BUILD)/ink-test

inkling-server: $(SERVER_OBJS) $(ENGINE_OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

$(BUILD)/httplib.o: vendor/cpp-httplib/httplib.cpp
	@mkdir -p $(BUILD)
	$(CXX) -std=c++17 -O2 -w -Ivendor -c $< -o $@

# the kernel source is one translation unit: main kernels, format codebooks, the kernels that use them
$(BUILD)/kernels_metal.h: src/kernels.metal src/quant_tables.metal src/kernels_quant.metal src/kernels_dspark.metal tools/embed_metal.py
	@mkdir -p $(BUILD)
	cat src/kernels.metal src/quant_tables.metal src/kernels_quant.metal src/kernels_dspark.metal > $(BUILD)/kernels_all.metal
	python3 tools/embed_metal.py $(BUILD)/kernels_all.metal $@

$(BUILD)/engine.o: src/engine.mm src/engine.h src/gguf.h src/quant_tables.h $(BUILD)/kernels_metal.h
	$(CXX) $(OBJCFLAGS) -c $< -o $@

$(BUILD)/%.o: src/%.cpp $(wildcard src/*.h)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/ink-test: $(BUILD)/ink_test.o $(ENGINE_OBJS)
	$(CXX) $^ $(LDFLAGS) -o $@

clean:
	rm -rf $(BUILD) inkling-server

.PHONY: all clean
