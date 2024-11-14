const fs = require('fs');
const pdfParse = require('pdf-parse');
const { NlpManager } = require('node-nlp');
const cosineSimilarity = require('cosine-similarity');
const readline = require('readline');
const { pipeline, AutoTokenizer, AutoModel } = require('@huggingface/transformers');

// Configuración del NlpManager
const manager = new NlpManager({ languages: ['es'], nlu: { log: false } });

// Función para extraer texto del PDF
const extractTextFromPDF = async (filePath) => {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdfParse(dataBuffer);
  return data.text;
};

// Función para dividir el texto en fragmentos
const splitTextIntoChunks = (text, chunkSize = 500) => {
  const chunks = [];
  let currentChunk = '';
  const words = text.split(' ');

  for (const word of words) {
    if ((currentChunk + word).length < chunkSize) {
      currentChunk += word + ' ';
    } else {
      chunks.push(currentChunk.trim());
      currentChunk = word + ' ';
    }
  }

  if (currentChunk) {
    chunks.push(currentChunk.trim());
  }

  return chunks;
};

// Función para cargar el modelo y el tokenizador
const loadModel = async () => {
  // Usar modelo en PyTorch (sin especificar ONNX)
  const tokenizer = await AutoTokenizer.from_pretrained('distilbert-base-uncased');
  const model = await AutoModel.from_pretrained('distilbert-base-uncased');
  return { tokenizer, model };
};

// Función para obtener embeddings con HuggingFace (utilizando PyTorch)
const getEmbedding = async (text, tokenizer, model) => {
  const encodedInput = tokenizer(text, { padding: true, truncation: true, return_tensors: 'pt' });
  const output = await model(encodedInput.input_ids);
  const embeddings = output.last_hidden_state[0].detach().numpy(); // Extracción del embedding
  return embeddings[0];  // Usamos el primer token (el [CLS]) para la representación
};

// Función para obtener embeddings de los fragmentos del texto
const getEmbeddingsForChunks = async (chunks, tokenizer, model) => {
  const embeddings = [];
  for (const chunk of chunks) {
    const embedding = await getEmbedding(chunk, tokenizer, model);
    embeddings.push(embedding);
  }
  return embeddings;
};

// Función para buscar la consulta en los fragmentos
const searchQuery = async (query, chunks, chunkEmbeddings, tokenizer, model) => {
  const queryEmbedding = await getEmbedding(query, tokenizer, model);

  // Calcular la similitud entre la consulta y los fragmentos
  const similarities = chunkEmbeddings.map((embedding, index) => ({
    index,
    similarity: cosineSimilarity(queryEmbedding, embedding),
  }));

  // Ordenar los fragmentos por similitud de mayor a menor
  similarities.sort((a, b) => b.similarity - a.similarity);

  // Imprimir los fragmentos más relevantes
  console.log(`Resultados para la consulta "${query}":`);
  similarities.slice(0, 5).forEach(result => {
    console.log(`Fragmento ${result.index} (Similitud: ${result.similarity.toFixed(4)}):`, chunks[result.index]);
  });
};

// Configuración del readline para recibir la entrada de la consulta desde la terminal
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Ruta al archivo PDF
const pdfPath = './swebok-v3.pdf'; // Asegúrate de que este archivo esté en la misma carpeta que index.js

// Función para ejecutar la búsqueda de consulta
const startSearch = () => {
  rl.question('¿Qué quieres saber sobre el SWEBOK V3?: ', (query) => {
    loadModel()
      .then(({ tokenizer, model }) => {
        extractTextFromPDF(pdfPath)
          .then(text => {
            const chunks = splitTextIntoChunks(text); // Aquí se define la variable `chunks`
            return getEmbeddingsForChunks(chunks, tokenizer, model).then(chunkEmbeddings => {
              return searchQuery(query, chunks, chunkEmbeddings, tokenizer, model);  // Pasa `chunks` correctamente
            });
          })
          .catch(err => console.error('Error al extraer el texto:', err))
          .finally(() => rl.close());
      })
      .catch(err => console.error('Error al cargar el modelo:', err));
  });
};

startSearch();
