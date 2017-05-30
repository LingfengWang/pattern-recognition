from keras.models import Model
from keras.layers import Input, Dense, Embedding, Reshape, GRU, merge, LSTM, Dropout, BatchNormalization, Activation
from keras.optimizers import RMSprop

def image_caption_model(vocab_size=11573, embedding_matrix=None, lang_dim=100,max_caplen=33, img_dim=4096, clipnorm=1):


    print('generating vocab_history model v5')
    # text: current word
    lang_input = Input(shape=(1,))
    x = Embedding(output_dim=lang_dim, input_dim=vocab_size, init='glorot_uniform', input_length=1)(lang_input)
    lang_embed = Reshape((lang_dim,))(x)
    # img
    img_input = Input(shape=(img_dim,))
    # text + img => GRU
    x = merge([img_input, lang_embed], mode='concat', concat_axis=-1)
    x = Reshape((1, lang_dim+img_dim))(x)
    x = GRU(128)(x)
    # predict next word
    out = Dense(vocab_size, activation='softmax')(x)
    model = Model(input=[img_input, lang_input], output=out)
    # choose objective and optimizer
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-3, clipnorm=clipnorm))
    return model
