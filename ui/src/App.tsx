import type { Component } from 'solid-js';

import logo from './logo.svg';
import styles from './App.module.css';

const App: Component = () => {
  return (
    <div class={styles.App}>
      <header class={styles.header}>
        <p>Diffusion Model Testing</p>
        <input type="text" />
        <button>Generate</button>
      </header>
    </div>
  );
};

export default App;
