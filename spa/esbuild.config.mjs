import * as esbuild from 'esbuild';
import { createHash } from 'crypto';
import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

const isWatch = process.argv.includes('--watch');

// Pin all Preact imports to a single instance — prevents the dual-Preact crash
// that occurs when file: deps (like superhot-ui) have their own node_modules/preact.
const preactAlias = {
  'preact':              resolve(__dirname, 'node_modules/preact'),
  'preact/hooks':        resolve(__dirname, 'node_modules/preact/hooks'),
  'preact/jsx-runtime':  resolve(__dirname, 'node_modules/preact/jsx-runtime'),
  '@preact/signals':     resolve(__dirname, 'node_modules/@preact/signals'),
};

const isProd = !isWatch;

const config = {
  entryPoints: ['src/index.jsx'],
  bundle: true,
  outfile: 'dist/bundle.js',
  format: 'esm',
  jsx: 'transform',
  jsxFactory: 'h',
  jsxFragment: 'Fragment',
  inject: ['./src/preact-shim.js'],
  loader: { '.jsx': 'jsx' },
  minify: isProd,
  sourcemap: isWatch,
  logLevel: 'info',
  alias: preactAlias,
};

const cssConfig = {
  entryPoints: ['src/index.css'],
  bundle: true,
  outfile: 'dist/bundle.css',
  minify: isProd,
  loader: { '.woff': 'file', '.woff2': 'file', '.ttf': 'file' },
  logLevel: 'info',
};

function injectVersionHash() {
  const js  = readFileSync('dist/bundle.js');
  const css = readFileSync('dist/bundle.css');
  const hash = createHash('sha256')
    .update(js).update(css)
    .digest('hex')
    .slice(0, 8);
  const html = readFileSync('index.html', 'utf8')
    .replace('bundle.css"', `bundle.css?v=${hash}"`)
    .replace('bundle.js"',  `bundle.js?v=${hash}"`);
  writeFileSync('dist/index.html', html);
  console.log(`Cache hash: ${hash}`);
}

if (isWatch) {
  const ctx = await esbuild.context(config);
  const cssCtx = await esbuild.context(cssConfig);
  await ctx.watch();
  await cssCtx.watch();
  console.log('esbuild watching for changes...');
} else {
  await esbuild.build(config);
  await esbuild.build(cssConfig);
  injectVersionHash();
}
