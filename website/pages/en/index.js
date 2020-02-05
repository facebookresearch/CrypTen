/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="projectLogo">
        <img src={props.img_src} alt="Project Logo" />
      </div>
    );

    const TitleLogo = props => (
        <img src={props.img_src} alt="CrypTen"/>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <TitleLogo img_src={`${baseUrl}img/crypten-logo-full.png`} />
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href="https://github.com/facebookresearch/crypten">GitHub</Button>
            <Button href="https://crypten.readthedocs.io/en/latest/">Docs</Button>
          </PromoSection>
          <PromoSection>
            <Button href="#try">Get Started</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );
    const Block2 = props => (
      <Container
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );
    const Tasks = () => (
      <div
        className="productShowcaseSection"
        style={{textAlign: 'left'}} id="tasks">
        <br /><br />
        <h2>How CrypTen Works</h2>
      </div>
    );

    const Tutorials = () => (
      <div
        style={{textAlign: 'center'}}>
        <p background="dark" id="tutorials">For more, checkout the
          <a href="https://github.com/facebookresearch/CrypTen#how-crypten-works">
           &nbsp;Tutorials
          </a>
        </p>
      </div>
    );

    const Installation = () => (
        <Block  background="dark" id="try" layout="twoColumn">
          {[
            {
              content: 'Please see the '+
              '[CrypTen Docs](https://github.com/facebookresearch/crypten#installing-crypten).'+
              '\n```bash\n\n pip install crypten\n```',
              title: 'Installation Script',
            },
            {
              content: 'CrypTensors encrypt data using familiar PyTorch '+
                        'syntax. For example: \n\n'+
                        '```python \n # PyTorch '+
                        '\n x = torch.tensor([1, 2, 3])\n '+
                        'y = torch.tensor([4, 5, 6])\n '+
                        'z = x + y\n ' +
                        '\n # CrypTen\n '+
                        'x_enc = crypten.cryptensor([1, 2, 3])\n '+
                        'y_enc = crypten.cryptensor([4, 5, 6])\n '+
                        'z_enc = x_enc + y_enc```',
              title: 'CrypTensors'
            },
          ]}
        </Block>
    );


    const Abstract = () => (
      <Block layout="fourColumn">
        {[
          {
            content:
              'CrypTen is a new framework built on PyTorch to facilitate ' +
              'research in secure and privacy-preserving machine learning. '+
              'CrypTen enables machine learning researchers, who '+
              'may not be cryptography experts, '+
              'to easily experiment with machine learning models '+
              'using secure computing techniques. '+
              'CrypTen lowers the barrier for machine learning researchers '+
              'by integrating with the common PyTorch API.',
            title: 'Secure Machine Learning',
          },
          {
            content: '',
            image: `${baseUrl}img/crypten-diagram.jpg`,
            imageAlign: 'bottom',
            title: '',
          },
        ]}
      </Block>
    );

    const Multiplication = () => (
      <Block layout="fourColumn">
        {[
          {
            content: '',
            image: `${baseUrl}img/crypten-multiplication.png`,
            imageAlign: 'bottom',
            title: '',
          },
          {
            content:
              (
              'MPC encrypts information by dividing data '+
              'between multiple parties, who can each perform calculations on '+
              'their share (in this example, 5 and 7) but are not able to '+
              'read the original data (12). <br> <br>'+
              'Each party then computes (\u201Cmultiply by 3\u201D). '+
              'When the outputs are combined, the result (36) is identical to '+
              'the result of performing the calculation on the data directly. '+
              'Since Party A and Party B do not know the end result (36) they '+
              'can not deduce the original data point (12).'),
            title: 'Multi-Party Compute: An Example',
          },
        ]}
      </Block>
    );

    const TaskOverview = () => (
      <Block layout="fourColumn">
        {[
          {
            content:
              'CrypTen currently implements a cryptographic method called '+
              'secure multiparty computation (MPC), and we expect to add '+
              'support for homomorphic encryption and secure enclaves in '+
              'futue releases. It works in the \u201Chonest but '+
              ' curious\u201D model (assumes the absence of malicious '+
              'and adversarial agents) that is used'+
              'frequently in cryptographic research, but additional '+
              'safeguards must be added before Crypten is ready to be used in '+
              'production settings ',
            title:  'Overview'
          },
          {
            image: `${baseUrl}img/crypten-pytorch-diagram.gif`,
            imageAlign: 'bottom',
          },
        ]}
      </Block>
    );

    const Filler = () => {
      return   <Block layout="fourColumn" background=""> {[]}</Block>;
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Abstract />
          <Installation />
          <Tutorials/>
          <Tasks />
          <TaskOverview />
          <Multiplication />
          <Filler />
        </div>
      </div>
    );
  }
}

module.exports = Index;
