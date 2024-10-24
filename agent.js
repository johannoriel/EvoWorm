// Constants
const kColor = 30;
const PI = Math.PI;

/**
 * Represents the environment where the car drives
 */
class Arena {
  constructor(width, height, canvas) {
    this.width = width;
    this.height = height;
    this.canvas = document.createElement("canvas");
    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.ctx.fillStyle = "#000";
    this.ctx.strokeStyle = "#000";

    //Optimization
    this.imageData = this.ctx.getImageData(0, 0, width, height);
    this.buffer = new Uint32Array(this.imageData.data.buffer);
  }

  clearAll() {
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.imageData = this.ctx.getImageData(0, 0, this.width, this.height);
    this.buffer = new Uint32Array(this.imageData.data.buffer);
  }

  // Check if a point collides with any obstacle
  collide(x, y) {
    if (
      typeof x !== "number" ||
      typeof y !== "number" ||
      isNaN(x) ||
      isNaN(y) ||
      x < 0 ||
      y < 0 ||
      x >= this.width ||
      y >= this.height
    ) {
      return false;
    }
    // Accès direct au buffer pour vérifier la collision
    return this.buffer[Math.floor(y) * this.width + Math.floor(x)] !== 0;
  }

  // Calculate depth until collision along a ray
  prof(x, y, theta, maxProf) {
    const c = Math.cos(theta);
    const s = Math.sin(theta);
    let result = 0;
    let currentX = x;
    let currentY = y;

    while (
      currentX >= 0 &&
      currentX < this.width &&
      currentY >= 0 &&
      currentY < this.height &&
      !this.collide(Math.floor(currentX), Math.floor(currentY)) &&
      result < maxProf
    ) {
      currentX += c;
      currentY += s;
      result++;
    }

    return result;
  }

  // Generate random obstacles
  generate(nbBoxes, boxHeight) {
    this.clearAll();

    // Draw border
    this.ctx.strokeRect(0, 0, this.width - 1, this.height - 1);

    // Generate random boxes
    for (let i = 0; i < nbBoxes; i++) {
      let x, y;
      do {
        x = Math.random() * (this.width - boxHeight);
        y = Math.random() * (this.height - boxHeight);
      } while (x < 40 && y < 40);

      this.ctx.fillRect(x, y, boxHeight, boxHeight);
    }

    // Mettre à jour le buffer après avoir dessiné
    this.imageData = this.ctx.getImageData(0, 0, this.width, this.height);
    this.buffer = new Uint32Array(this.imageData.data.buffer);
  }
}

/**
 * Base class for the car that can drive in the arena
 */
class Agent {
  constructor(
    arena,
    visionX,
    visionY,
    factorTheta,
    factorDepth,
    startSpeed,
    vMin,
    vMax,
    frott,
  ) {
    this.arena = arena;
    this.visionX = visionX;
    this.visionY = visionY;
    this.factorTheta = factorTheta;
    this.factorDepth = factorDepth;
    this.vMin = vMin;
    this.vMax = vMax;
    this.frott = frott;

    // Initialize state
    this.init(startSpeed);

    // Vision array
    this.view = new Float32Array(visionX * visionY);
  }

  init(speed) {
    this.x = 10;
    this.y = 10;
    this.v = speed;
    this.theta = PI / 4;
    this.vTheta = 0;
    this.a = 0;
    this.collision = 0;
    this.distance = 0;
  }

  setAccel(acc) {
    this.a = acc;
  }

  setRotation(rot) {
    this.vTheta = Math.max(-PI / 3, Math.min(PI / 3, rot));
  }

  iterate(ctx, dt) {
    const oldX = Math.floor(this.x);
    const oldY = Math.floor(this.y);

    // Update orientation
    this.theta += this.vTheta * dt;
    this.theta = this.theta < 0 ? this.theta + 2 * PI : this.theta;
    this.theta = this.theta > 2 * PI ? this.theta - 2 * PI : this.theta;

    // Update velocity
    this.v += (this.a - this.frott * this.v) * dt;
    this.v = Math.max(this.vMin, Math.min(this.vMax, this.v));

    // Update position
    this.x += this.v * Math.cos(this.theta) * dt;
    this.y += this.v * Math.sin(this.theta) * dt;

    // Check collision
    this.collision = this.arena.collide(Math.floor(this.x), Math.floor(this.y));
    if (this.collision) {
      this.x = oldX;
      this.y = oldY;
    }

    // Draw movement trail if context provided
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(oldX, oldY);
      ctx.lineTo(this.x, this.y);
      ctx.strokeStyle = `rgb(${kColor + 64}, ${kColor + 64}, ${kColor + 64})`;
      ctx.stroke();
    }

    // Update distance
    this.distance +=
      this.v * dt * (this.x / this.arena.width) * (this.y / this.arena.height);
  }

  vision(ctx) {
    const step = (PI / 300) * this.factorTheta;
    let tTheta = this.theta - (step * this.visionX) / 2;

    for (let i = 0; i < this.visionX; i++, tTheta += step) {
      const color =
        i === Math.floor(this.visionX / 2) ? kColor - 3 : kColor + 1;
      const dist = this.arena.prof(
        this.x,
        this.y,
        tTheta,
        this.visionY * this.factorDepth,
      );

      // Draw vision rays if context provided
      if (ctx) {
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(
          this.x + dist * Math.cos(tTheta),
          this.y + dist * Math.sin(tTheta),
        );
        ctx.strokeStyle = `rgb(${color}, ${color}, ${color})`;
        ctx.stroke();
      }

      // Update vision array
      for (let j = 0; j < this.visionY; j++) {
        this.view[i + j * this.visionX] = 0;
      }

      const ps = Math.floor(dist / this.factorDepth);
      for (let j = ps; j < this.visionY; j++) {
        this.view[i + j * this.visionX] = 1;
      }
    }
  }
}

/**
 * Extended car class with different vision method
 */
class MapAgent extends Agent {
  constructor(
    arena,
    visionX,
    visionY,
    factorTheta,
    factorDepth,
    startSpeed,
    vMin,
    vMax,
    frott,
  ) {
    super(
      arena,
      visionX,
      visionY,
      factorTheta,
      factorDepth,
      startSpeed,
      vMin,
      vMax,
      frott,
    );

    // Create viewer canvas
    this.viewer = document.createElement("canvas");
    this.viewer.width = visionX;
    this.viewer.height = visionY;
    this.viewerCtx = this.viewer.getContext("2d");
  }

  vision(ctx) {
    const s = Math.sin(this.theta) * this.visionX * this.factorTheta;
    const c = Math.cos(this.theta) * this.visionX * this.factorTheta;

    // Calculate vision box corners
    const Ax = Math.floor(this.x - s / 2);
    const Ay = Math.floor(this.y + c / 2);
    const Bx = Math.floor(this.x + s / 2);
    const By = Math.floor(this.y - c / 2);
    const Dx = Math.floor(
      Ax + Math.cos(this.theta) * this.visionY * this.factorDepth - s,
    );
    const Dy = Math.floor(
      Ay + Math.sin(this.theta) * this.visionY * this.factorDepth + c,
    );
    const Cx = Math.floor(
      Bx + Math.cos(this.theta) * this.visionY * this.factorDepth + s,
    );
    const Cy = Math.floor(
      By + Math.sin(this.theta) * this.visionY * this.factorDepth - c,
    );

    // Draw vision box if context provided
    if (ctx) {
      ctx.beginPath();
      ctx.moveTo(Ax, Ay);
      ctx.lineTo(Bx, By);
      ctx.lineTo(Cx, Cy);
      ctx.lineTo(Dx, Dy);
      ctx.closePath();
      ctx.strokeStyle = `rgb(${kColor + 32}, ${kColor + 32}, ${kColor + 32})`;
      ctx.stroke();
    }

    // Transform view into vision array
    // This is a simplified version as the original used complex mapping
    this.viewerCtx.clearRect(0, 0, this.visionX, this.visionY);
    this.viewerCtx.save();
    this.viewerCtx.transform(
      this.visionX / (Bx - Ax),
      0,
      0,
      this.visionY / (Cy - By),
      -Ax * (this.visionX / (Bx - Ax)),
      -By * (this.visionY / (Cy - By)),
    );
    this.viewerCtx.drawImage(this.arena.canvas, 0, 0);
    this.viewerCtx.restore();

    // Update vision array
    const imageData = this.viewerCtx.getImageData(
      0,
      0,
      this.visionX,
      this.visionY,
    );
    for (let i = 0; i < this.visionX; i++) {
      for (let j = 0; j < this.visionY; j++) {
        const idx = (i + j * this.visionX) * 4;
        this.vision[i + j * this.visionX] =
          imageData.data[idx + 3] === 0 ? 1 : 0;
      }
    }
  }
}
