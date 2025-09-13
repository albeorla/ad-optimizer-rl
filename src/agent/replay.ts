export interface Transition<S extends readonly number[]> {
  s: S;
  aIdx: number;
  r: number;
  sp: S;
  done: boolean;
}

export class ReplayBuffer<S extends readonly number[]> {
  private buf: Transition<S>[] = [];
  private next = 0;

  constructor(private capacity: number) {
    if (capacity <= 0) throw new Error("ReplayBuffer capacity must be > 0");
  }

  size(): number {
    return this.buf.length;
  }

  push(t: Transition<S>): void {
    if (this.buf.length < this.capacity) {
      this.buf.push(t);
    } else {
      this.buf[this.next] = t;
    }
    this.next = (this.next + 1) % this.capacity;
  }

  sample(batchSize: number): Transition<S>[] {
    const n = Math.min(batchSize, this.buf.length);
    const out: Transition<S>[] = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * this.buf.length);
      out.push(this.buf[idx]!);
    }
    return out;
  }
}

