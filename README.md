<!-- PROJECT LOGO -->
<p align="center">
  <h1 align="center">AmoebaNet Evolver - PyTorch</h3>

  <p align="center">
    System to evolve the structure of AmoebaNet-D to attempt to improve performance by making mutations to the cell operations used. 
    <br />
    <a href="https://github.com/whiteio/AmoebaNet-Evolver"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="https://github.com/whiteio/AmoebaNet-Evolver/issues">Report Bug</a>
    ·
    <a href="https://github.com/whiteio/AmoebaNet-Evolver/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

This project can be used to evolve the structure of AmoebaNet-D to improve classification performance, as it was created through evolution using the CIFAR-10 dataset which may not be as effective in alternative domains such as chest infection classification. The dataset currently used is ChestX-ray14 and the models are configured for multi-label image classification. The average AUROC score is currently used as the metric for deciding which models perform best during each cycle.

## Key Details
* Each model is created by mutating the best performing model from the previous cycle
* A cycle involves mutating the population of models, training for a user definable number of epochs and evaluating 
* AmoebaNet-D is initial structure used
* Mutations occur to the operations of the normal and reduction cells
* Each cycle one operation mutation occurs per model
* Operations are randomly picked from a subset of NAS search space (operations already used in AmoebaNet-A,B,C and D)
* Results are output to log file containing the evaluation scores obtained for each model, along with the models normal and reduction operations for reproducability



### Built With

* PyTorch
* [GPipe](https://github.com/kakaobrain/torchgpipe/tree/master/benchmarks/models/amoebanet) - Pytorch implementation of AmoebaNet-D
* Modified hyperparameter tuning code to be used for evolution, obtained from [here](https://github.com/voiler/PopulationBasedTraining)



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC) dataset (hopefully the evolver will be abstracted soon to apply it to different domains easily)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/whiteio/AmoebaNet-Evolver.git
   ```
2. Install required packages
   ```sh
   pip install -r requirements.txt
   ```



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/whiteio/AmoebaNet-Evolver/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Christopher White - chris@cwhite.io

Project Link: [https://github.com/whiteio/AmoebaNet-Evolver](https://github.com/whiteio/AmoebaNet-Evolver)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [GPipe AmoebaNet-D implementation](https://github.com/kakaobrain/torchgpipe)
* [Population based training](https://github.com/voiler/PopulationBasedTraining)





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/whiteio/AmoebaNet-Evolver/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
