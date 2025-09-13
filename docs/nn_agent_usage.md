## NN Agent Usage (DQN)

- Select NN agent: `npm start -- --agent=nn`
- Flags: `--episodes`, `--batchSize`, `--gamma`, `--lr`, `--trainFreq`, `--targetSync`, `--replayCap`, `--epsilonStart`, `--epsilonMin`, `--epsilonDecay`
- Save/Load: `await agent.save('model.json')` / `await agent.load('model.json')`
- Backend: default `@tensorflow/tfjs`; consider `@tensorflow/tfjs-node` for speed
- References: `src/agent/dqnAgentNN.ts`, `src/agent/nn/qnet.ts`, `src/agent/encoding.ts`
