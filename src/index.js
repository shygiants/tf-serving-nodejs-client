import path from 'path';
import grpc from 'grpc';
import jpeg from 'jpeg-js';

const PROTO_PATH = path.join(__dirname, '../protos/prediction_service.proto');
const SIGNATURE_NAME = 'predict_images';

export default class TFServing {
  /**
   * 
   * @param {string} connection 
   */
  static connect(connection) {
    const tfServing = grpc.load(PROTO_PATH).tensorflow.serving;
    const client = new tfServing.PredictionService(
      connection, 
      grpc.credentials.createInsecure());
    return new Client(client);
  }
}

class Client {
  /**
   * 
   * @param {grpc.Client} client 
   */
  constructor(client) {
    this.client = client;
  }

  /**
   * 
   * @param {Buffer|Array<Buffer>} images
   * @param {string} name 
   * @param {{decode: boolean, version: number}} [option]
   * @return {Promise}
   */
  predict(images, name, {decode, version}) {
    decode = decode || false;
    version = version || 1;
    let imagesMsg;
    if (decode) {
      const decoded = jpeg.decode(images, true);
      imagesMsg = {
        dtype: 'DT_INT32',
        tensor_shape: {
          dim: [
            { size: decoded.height },
            { size: decoded.width },
            { size: 3 },
          ]
        },
        int_val: Array.from(decoded.data)
      }
    } else {
      if (!(images instanceof Array)) {
        images = [images];
      }
      imagesMsg = {
        dtype: 'DT_STRING',
        tensor_shape: {
          dim: {
            size: images.length
          }
        },
        string_val: images
      }
    }

    
    const predictRequest = {
      model_spec: {
        name,
        signature_name: SIGNATURE_NAME,
        version: {
          value: version
        }
      },
      inputs: {
        images: imagesMsg
      }
    }

    return new Promise((resolve, reject) => {
      this.client.predict(predictRequest, (err, res) => {
        if (err) return reject(err);

        const {scores} = res.outputs;
        const {float_val, tensor_shape} = scores;

        if (decode) {
          resolve(float_val);
        } else {
          const numLabels = Number(tensor_shape.dim[1].size);
          const vals = [...Array(images.length).keys()].map(idx => 
            float_val.slice(numLabels * idx, numLabels * (idx + 1))
          )
          resolve(vals);
        }
      });
    });
  }
}

if (require.main === module) {
  // Run directly from Node.js
  const config = require('config');
  const fs = require('fs');
  const argmax = require( 'compute-argmax' );
  
  // Load image
  const image = fs.readFileSync(config.get('testImagePath'));

  // Predict
  const client = TFServing.connect(config.get('connection'));
  client
    .predict(image, 
             config.get('modelName'), 
             { version: config.get('version') })
    .then(vals => {
      vals.forEach(val => {
        console.log(`Scores: ${val}`);
        console.log(`Argmax: ${argmax(val)}`);
      });
    })
    .catch(console.error);
}