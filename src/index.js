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
   * @param {Buffer} image
   * @param {string} name 
   */
  predict(image, name) {
    const decoded = jpeg.decode(image, true);
    const predictRequest = {
      model_spec: {
        name,
        signature_name: SIGNATURE_NAME
      },
      inputs: {
        images: {
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
      }
    }

    return new Promise((resolve, reject) => {
      this.client.predict(predictRequest, (err, res) => {
        if (err) return reject(err);
        resolve(res.outputs.scores.float_val);
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
    .predict(image, config.get('modelName'))
    .then(vals => {
      console.log(`Scores: ${vals}`);
      console.log(`Argmax: ${argmax(vals)}`);
    })
    .catch(console.error);
}