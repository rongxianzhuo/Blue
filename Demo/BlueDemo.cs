using UnityEngine;
using UnityEngine.UI;

namespace Blue.Samples
{

    public class BlueDemo : MonoBehaviour
    {
        
        public Text infoText;

        private readonly BaseSample _sample = new MnistSample();

        private void Awake()
        {
            StartCoroutine(_sample.Run(5));
        }

        private void Update()
        {
            infoText.text = _sample.Info;
        }

        private void OnDestroy()
        {
            _sample.Stop();
        }
    }

}