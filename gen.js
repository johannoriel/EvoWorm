class Creature {
  constructor() {
    this.score = 0;
    this.time = 0;
    this.alive = true;
    this.muted = true;
    this.animating = false;
    this.paused = false;
    this.canvas = null;
    this.porsche = null; // Instance de Porsche spécifique à cette creature

    // Initialize neural network
    this.initNetwork();
    this.initPorsche();
    this.animationFrameId = null;
    this.processScore = this.processScore.bind(this);
    this.animateLoop = this.animateLoop.bind(this);
    this._boundAnimateLoop = this.animateLoop.bind(this);
    this.lastFrame = performance.now();
    this.frameInterval = 1000 / 60; // 60 FPS
  }

  reset() {
    this.time = 0;
    this.alive = true;
    this.score = 0;
    this.stopAnimation();
    this.canvas = null;
    if (this.porsche) {
      this.porsche.init(StartSpeed);
    }
  }

  initNetwork() {
    // Initialize layers with proper dimensions
    const numInputs = VisionX * VisionY + 2 * FeedBack + FeedBackSpeed;
    this.entries = new Layer(numInputs);
    this.outputs = new Layer(2);

    // Initialize network
    this.net = new NeuralNet(this.entries, this.outputs);

    // Initialize output layer weights
    this.outputs.connect(this.entries);
    this.outputs.initRandom(0.2);
  }

  initPorsche() {
    this.porsche = new Porsche(
      scene,
      VisionX,
      VisionY,
      FactorTheta,
      FactorDepth,
      StartSpeed,
      MinSpeed,
      MaxSpeed,
      kFrott,
    ); // Supposant que vous avez une classe Porsche
    this.porsche.init(StartSpeed);
  }

  copy(creature) {
    this.net.copy(creature.net);
    this.score = creature.score;
    this.time = creature.time;
    this.alive = creature.alive;
    this.muted = creature.muted;
  }

  mute(percent, factor) {
    this.muted = true;
    this.net.mute(percent, factor);
    this.time = 0;
    this.alive = true;
    this.score = 0;
  }

  median() {
    return this.net.median();
  }

  animateStep(display, dt) {
    if (!this.alive || this.time >= TimeLimit) {
      //console.log("Not alive anymore");
      return false;
    }
    if (!this.porsche) {
      this.initPorsche();
    }

    this.time++;

    if (display && this.canvas) {
      // Clear and draw background
      const ctx = this.canvas.getContext("2d");
      ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.drawImage(scene.canvas, 0, 0);

      // Draw and update Porsche
      this.porsche.iterate(ctx, dt);
      this.porsche.vision(ctx);
      //console.log("animate Step "+this.time);

      // Check collision
      this.alive = !this.porsche.collision;
    } else {
      this.porsche.iterate(null, dt);
      this.porsche.vision(null);
      this.alive = !this.porsche.collision;
    }

    // Update neural network inputs from vision
    for (let i = 0; i < VisionX * VisionY; i++) {
      this.entries.neurons[i].output = this.porsche.view[i];
    }

    // Add feedback if enabled
    for (let i = 0; i < FeedBack; i++) {
      const baseIndex = VisionX * VisionY;
      this.entries.neurons[baseIndex + i].output = this.outputs[0].output;
      this.entries.neurons[baseIndex + FeedBack + i].output =
        this.outputs[1].output;
    }

    // Add speed feedback if enabled
    if (FeedBackSpeed > 0) {
      const baseIndex = VisionX * VisionY + 2 * FeedBack;
      for (let i = 0; i < FeedBackSpeed; i++) {
        this.entries.neurons[baseIndex + i].output = 0;
      }

      const speedIndex = Math.floor((porsche.v * FeedBackSpeed) / MaxSpeed);
      if (speedIndex >= 0 && speedIndex < FeedBackSpeed) {
        this.entries.neurons[baseIndex + speedIndex].output = 1;
      }
    }

    // Process network and update car controls
    this.net.process();
    this.porsche.setAccel(this.outputs.neurons[0].output * AccMax);
    this.porsche.setRotation(this.outputs.neurons[1].output);

    return true;
  }

  animateLoop(timestamp) {
    //console.log('animateLoop called', timestamp);
    if (this.paused) {
      //this.animationFrameId = requestAnimationFrame(this._boundAnimateLoop);
      return;
    }
    const shouldContinue = this.animateStep(true, 0.01);
    const elapsed = timestamp - this.lastFrame;

    if (elapsed >= this.frameInterval) {
      this.lastFrame = timestamp - (elapsed % this.frameInterval);

      if (!shouldContinue) {
        //console.log("Should not continue");
        this.stopAnimation();
        this.score = this.porsche.distance;
        this.muted = false;
        //console.log(`Animation finished - Score: ${this.score.toFixed(2)}, Time: ${this.time}`);
        return;
      }
      this.animationFrameId = requestAnimationFrame(this._boundAnimateLoop);
    }
    //console.log("Looping animation loop");
    this.animationFrameId = requestAnimationFrame(this._boundAnimateLoop);
  }

  processScore(display, canvas, dt) {
    if (!display && !this.muted) return;

    // Reset initial conditions
    this.time = 0;
    this.alive = true;

    // Initialize or reset Porsche
    if (!this.porsche) {
      this.initPorsche();
    } else {
      this.porsche.init(StartSpeed);
    }

    if (display) {
      // Store canvas reference and start animation loop
      this.canvas = canvas;
      this.animating = true;
      this.paused = false;
      this.animationFrameId = requestAnimationFrame(this._boundAnimateLoop);
    } else {
      // Run simulation without animation for training
      while (this.time !== TimeLimit && this.alive) {
        this.animateStep(false, dt);
      }
      this.score = this.porsche.distance;
      this.muted = false;
    }
  }

  stopAnimation() {
    //console.log('Stopping animation');
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.animating = false;
    this.canvas = null;
  }

  pauseAnimation() {
    this.paused = !this.paused;
  }
}

class Population {
  constructor(size, dt) {
    this.nbCreatures = size;
    this.creatures = Array(size)
      .fill(null)
      .map(() => new Creature());
    this.scores = Array(size)
      .fill(null)
      .map((_, i) => ({ indice: i, score: 0 }));
    this.generation = 0;
    this.best = 0;
    this.time = 0;

    this.newMap();
    this.iterate(dt);
    this.sort();
  }

  sort() {
    for (let i = 0; i < this.nbCreatures; i++) {
      this.scores[i] = {
        indice: i,
        score: this.creatures[i].score,
      };
    }

    this.scores.sort((a, b) => b.score - a.score);

    this.best = this.creatures[this.scores[0].indice].score;
    this.time = this.creatures[this.scores[0].indice].time;
  }

  mute(muteRatio, muteFactor) {
    const nbFirst = Math.floor(this.nbCreatures / 10);
    const nbMid = Math.floor(this.nbCreatures / 2);

    // Clone and mutate best performers
    for (let i = 0; i < nbFirst; i++) {
      const targetIndex1 = this.nbCreatures - 2 * i - 1;
      const targetIndex2 = this.nbCreatures - 2 * i - 2;

      if (targetIndex1 >= 0) {
        this.creatures[this.scores[targetIndex1].indice].copy(
          this.creatures[this.scores[i].indice],
        );
        this.creatures[this.scores[targetIndex1].indice].mute(
          muteRatio,
          muteFactor,
        );
      }

      if (targetIndex2 >= 0) {
        this.creatures[this.scores[targetIndex2].indice].copy(
          this.creatures[this.scores[i].indice],
        );
        this.creatures[this.scores[targetIndex2].indice].mute(
          muteRatio,
          muteFactor,
        );
      }
    }

    // Handle middle section
    for (let i = nbFirst; i < nbMid - nbFirst; i++) {
      const targetIndex = nbMid + i - nbFirst;
      if (targetIndex < this.nbCreatures) {
        this.creatures[this.scores[targetIndex].indice].copy(
          this.creatures[this.scores[i].indice],
        );
        this.creatures[this.scores[targetIndex].indice].mute(
          muteRatio,
          muteFactor,
        );
      }
    }
  }

  process(muteRatio, muteFactor, dt) {
    this.mute(muteRatio, muteFactor);
    this.iterate(dt);
    this.sort();
    this.generation++;
  }

  iterate(dt) {
    for (let i = 0; i < this.nbCreatures; i++) {
      this.creatures[i].processScore(false, null, dt);
      /*console.log(
        this.creatures[i].median(),
        " : ",
        this.creatures[i].porsche.distance,
        " : ",
        this.creatures[i].time,
      );*/
    }
  }

  getBest() {
    return this.creatures[this.scores[0].indice];
  }

  newMap() {
    // Arrêter les animations existantes
    for (let i = 0; i < this.nbCreatures; i++) {
      this.creatures[i].stopAnimation();
    }

    scene.generate(NbBoxes, BoxH);
    for (let i = 0; i < this.nbCreatures; i++) {
      this.creatures[i].muted = true;
    }
    this.iterate(0.1);
    this.sort();
  }
}
