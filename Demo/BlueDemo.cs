using UnityEngine;
using UnityEngine.UI;

namespace Blue.Samples
{

    public class BlueDemo : MonoBehaviour
    {
        
        public Text infoText;

        private readonly MnistSample _sample = new MnistSample();

        private void Awake()
        {
            StartCoroutine(_sample.Run());
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