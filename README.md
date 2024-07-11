# Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detectors

Official code release for [Out-of-Bounding-Box Triggers: A Stealthy Approach to Cheat Object Detectors](https://github.com/linToTao/Out-of-bbox-attack).

<p align='center'>
  <b>
    <a href="https://github.com/linToTao/Out-of-bbox-attack">Paper</a>
    |
    <a href="https://github.com/linToTao/Out-of-bbox-attack">Code</a> 
  </b>
</p> 
  <p align='center'>
    <img src='static/out-of-bbox-attack.png' width='1000'/>
  </p>

**Abstract**: In recent years, the study of adversarial robustness in object detection systems, particularly those based on deep neural networks (DNNs), has become a pivotal area of research. Traditional physical attacks targeting object detectors, such as adversarial patches and texture manipulations, directly manipulate the surface of the object. While these methods are effective, their overt manipulation of objects may draw attention in real-world applications. To address this, this paper introduces a more subtle approach: an inconspicuous adversarial trigger that operates outside the bounding boxes, rendering the object undetectable to the model. We further enhance this approach by proposing the <mark>Feature Guidance (FG)</mark> technique and the <mark>Universal Auto-PGD (UAPGD)</mark> optimization strategy for crafting high-quality triggers. The effectiveness of our method is validated through extensive empirical testing, demonstrating its high performance in both digital and physical environments.
