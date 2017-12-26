# Boring Detector

A state-of-the-art detector of Boring hats. Developed at MIT to solve the global challenge (implicitly) posed by Elon
Musk when he declared that 10 Boring hat owners will be awared a Boring tour. Given that 50,000 hats (current sale
estimates) cover less than 0.001% of 7.3 billion heads on Earth, an automated method for Boring hat detection is needed.

The Boring detector is a fine-tuned deep neural network model based on a [Keras implementation][3] of a
[RetinaNet object detection architecture][4]. The following are some results. The Boring Score is from 0 to 1. Anything
above a score of 0.5 is considered Boring.

![Boring Hat](https://github.com/lexfridman/boring-detector/raw/master/showcase/boring-hat-fast-detected.gif)

---

![Falcon Heavy](https://github.com/lexfridman/boring-detector/raw/master/showcase/falcon-heavy-detected.gif)

---

![Holiday Minions](https://github.com/lexfridman/boring-detector/raw/master/showcase/holiday-minions-detected.gif)

---

![Cat Monkey](https://github.com/lexfridman/boring-detector/raw/master/showcase/cat-monkey-detected.gif)


## Comparison to Competitors

|                        |   [Boring Detector][1]  |  [Hot Dog Detector][2]  |
| ---------------------- |:-----------------------:|:-----------------------:|
| Object localization    |            x            |                         |
| Boring hat detection   |            x            |                         |
| Hot dog classification |        (in beta)        |            x            |

[1]: https://lex.mit.edu/boring
[2]: https://play.google.com/store/apps/details?id=com.seefoodtechnologies.nothotdog&hl=en
[3]: https://github.com/fizyr/keras-retinanet 
[4]: https://arxiv.org/abs/1708.02002

## Instructions

If you would like an image or video of your own to be verified by the Boring detector, post it on Instagram or Twitter with
hashtag __#boringdetector__ (boring for lex). I'll then make links to processed versions of these images/videos here...

If you would like to run the network on your own machine, I'll post detailed instructions here soon.

## License

Boring Detector uses Apache License 2.0. See
[LICENSE](https://github.com/lexfridman/boring-detector/blob/master/LICENSE) for more details.
