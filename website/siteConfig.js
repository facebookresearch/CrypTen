/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

const siteConfig = {
  title: 'CrypTen', // Title for your website.
  tagline: 'A research tool for secure machine learning in PyTorch',
  url: 'https://facebookresearch.github.io/CrypTen/', // Your website URL
  baseUrl: '', // Base URL for your project */

  // Used for publishing and more
  projectName: 'crypten',
  cname: 'crypten.ai',
  organizationName: 'facebookresearch',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    {
      href: 'https://github.com/facebookresearch/crypten',
      label: 'GitHub',
      external: true
    },
    {
     href: 'https://crypten.readthedocs.io/en/latest/',
     label: 'Docs',
     external: true
    },
  ],
  /* path to images for header/footer */
  headerIcon: 'img/crypten-logo-full-white.png',
  footerIcon: 'img/crypten-logo-full-white.png',
  favicon: 'img/crypten-icon.png',

  /* Colors for website */
  colors: {
    // primaryColor: '#2C9848',
    primaryColor: '#100833',
    secondaryColor: '#7850D7',
  },


  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright \u{00A9} ${new Date().getFullYear()} Facebook`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'github',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: [
  'https://buttons.github.io/buttons.js',
  'https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.0/clipboard.min.js',
  'js/code-block-buttons.js',
  '/js/redirect.js',
],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/undraw_online.svg',
  twitterImage: 'img/undraw_tweetstorm.svg',
};

module.exports = siteConfig;
