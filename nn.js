// Neural Network implementation in JavaScript
// Converted from C++ code originally by LEBAB (25/11/95)

class Neuron {
    constructor() {
        this.connections = []; // Array of {neuron, weight}
        this.output = 0;
        this.input = 0;
        this.di = 0; // result of quadratical derivation
    }

    static transfer(value) {
        return Math.tanh(value);
    }

    static transferPrime(value) {
        if (Math.abs(value) > 100) return 0;
        return 1 / Math.pow(Math.cosh(value), 2);
    }

    setConnections(count) {
        this.connections = new Array(count).fill(null).map(() => ({
            neuron: null,
            weight: 0
        }));
    }

    process() {
        this.input = this.connections.reduce((sum, conn) =>
            sum + conn.neuron.output * conn.weight, 0);
        this.output = Neuron.transfer(this.input);
    }

    initRandom(amplitude) {
        this.connections.forEach(conn => {
            conn.weight = (Math.random() - 0.5) * amplitude;
        });
    }

    initConst(c) {
        this.connections.forEach(conn => {
            conn.weight = c;
        });
    }

    copy(neuron) {
        if (this.connections.length !== neuron.connections.length) {
            throw new Error("Neurons must have same number of connections");
        }
        this.connections.forEach((conn, i) => {
            conn.weight = neuron.connections[i].weight;
        });
    }

    reproduction(neuron, p) {
        if (this.connections.length !== neuron.connections.length) {
            throw new Error("Neurons must have same number of connections");
        }
        this.connections.forEach((conn, i) => {
            if (Math.random() < p) {
                conn.weight = neuron.connections[i].weight;
            }
        });
    }

    mute(percent, factor) {
        const median = this.median();
        this.connections.forEach(conn => {
            if (Math.random() < percent) {
                conn.weight += (Math.random() - 0.5) * factor * median;
            }
        });
    }

    median() {
        return this.connections.reduce((sum, conn) =>
            sum + Math.abs(conn.weight), 0) / this.connections.length;
    }

    retroPropagate(newDi, gradient) {
        const factor = newDi * gradient;
        this.di = newDi;
        this.connections.forEach(conn => {
            conn.weight -= factor * conn.neuron.output;
        });
    }
}

class Layer {
    constructor(neuronCount = 0) {
        this.neurons = Array(neuronCount).fill(null).map(() => new Neuron());
    }

    process() {
        this.neurons.forEach(neuron => neuron.process());
    }

    connect(layer) {
        this.neurons.forEach(neuron => {
            neuron.setConnections(layer.neurons.length);
            neuron.connections.forEach((conn, j) => {
                conn.neuron = layer.neurons[j];
            });
        });
    }

    initRandom(amplitude) {
        this.neurons.forEach(neuron => neuron.initRandom(amplitude));
    }

    initConst(c) {
        this.neurons.forEach(neuron => neuron.initConst(c));
    }

    mute(percent, factor) {
        this.neurons.forEach(neuron => neuron.mute(percent, factor));
    }

    copy(layer) {
        if (this.neurons.length !== layer.neurons.length) {
            throw new Error("Layers must have same number of neurons");
        }
        this.neurons.forEach((neuron, i) => neuron.copy(layer.neurons[i]));
    }

    reproduction(layer, percent) {
        if (this.neurons.length !== layer.neurons.length) {
            throw new Error("Layers must have same number of neurons");
        }
        this.neurons.forEach((neuron, i) =>
            neuron.reproduction(layer.neurons[i], percent));
    }

    median() {
        return this.neurons.reduce((sum, neuron) =>
            sum + neuron.median(), 0) / this.neurons.length;
    }

    retroOutput(gradient, ideal) {
        if (this.neurons.length !== ideal.neurons.length) {
            throw new Error("Layers must have same size for retropropagation");
        }
        this.neurons.forEach((neuron, i) => {
            const si = neuron.output;
            const yi = ideal.neurons[i].output;
            const ii = neuron.input;
            const ti = Neuron.transferPrime(ii);
            const di = 2 * (si - yi) * ti;
            neuron.retroPropagate(di, gradient);
        });
    }

    retroHidden(gradient, nextLayer) {
        this.neurons.forEach((neuron, i) => {
            let di = 0;
            const ti = Neuron.transferPrime(neuron.input);
            nextLayer.neurons.forEach(nextNeuron => {
                const connectionToThis = nextNeuron.connections.find(
                    conn => conn.neuron === neuron
                );
                if (connectionToThis) {
                    di += nextNeuron.di * connectionToThis.weight * ti;
                }
            });
            neuron.retroPropagate(di, gradient);
        });
    }

    quadraticError(ideal) {
        if (this.neurons.length !== ideal.neurons.length) {
            throw new Error("Layers must have same size for error calculation");
        }
        return this.neurons.reduce((sum, neuron, i) =>
            sum + Math.pow(neuron.output - ideal.neurons[i].output, 2), 0);
    }
}

class NeuralNet {
    constructor(entries, outputs) {
        this.entries = entries;
        this.outputs = outputs;
        this.layers = []; // Hidden layers between entries and outputs
    }

    addLayer(layer) {
        this.layers.push(layer);
    }

    process() {
        this.layers.forEach(layer => layer.process());
        this.outputs.process();
    }

    mute(percent, factor) {
        this.layers.forEach(layer => layer.mute(percent, factor));
        this.outputs.mute(percent, factor);
    }

    reproduction(net, percent) {
        if (this.layers.length !== net.layers.length) {
            throw new Error("Networks must have same number of layers");
        }
        this.layers.forEach((layer, i) =>
            layer.reproduction(net.layers[i], percent));
        this.outputs.reproduction(net.outputs, percent);
    }

    copy(net) {
        if (this.layers.length !== net.layers.length) {
            throw new Error("Networks must have same number of layers");
        }
        this.layers.forEach((layer, i) => layer.copy(net.layers[i]));
        this.outputs.copy(net.outputs);
    }

    median() {
        const layerMedians = this.layers.reduce((sum, layer) =>
            sum + layer.median(), 0);
        return (layerMedians + this.outputs.median()) / (this.layers.length + 1);
    }

    retroPropagate(gradient, ideal) {
        this.outputs.retroOutput(gradient, ideal);

        if (this.layers.length > 0) {
            const reversedLayers = [...this.layers].reverse();

            // Retropropagate through hidden layers
            reversedLayers[0].retroHidden(gradient, this.outputs);
            for (let i = 1; i < reversedLayers.length; i++) {
                reversedLayers[i].retroHidden(gradient, reversedLayers[i-1]);
            }

            // Retropropagate to entries
            this.entries.retroHidden(gradient, this.layers[0]);
        }
    }

    quadraticError(ideal) {
        return this.outputs.quadraticError(ideal);
    }
}
