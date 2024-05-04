$(document).ready(function(){
    $('#uploadForm').on('submit', function(e){
        e.preventDefault();
        $('#uploadStatus').html('<p>Status: Processing...</p>');
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: new FormData(this),
            contentType: false,
            cache: false,
            processData:false,
            success: function(response){
                $('#uploadStatus').html('<p>Status: '+response.status+'</p>');
            }
        });
    });
});


(function () {
    var Message;
    Message = function (arg) {
        this.text = arg.text, this.message_side = arg.message_side;
        this.draw = function (_this) {
            return function () {
                var $message;
                $message = $($('.message_template').clone().html());
                $message.addClass(_this.message_side === 'user' ? 'user-message' : 'bot-message')
                        .find('.text').html(_this.text);
                $('.messages').append($message);
                return setTimeout(function () {
                    return $message.addClass('appeared');
                }, 0);
            };
        }(this);
        return this;
    };

    $(function () {
        var getMessageText, sendMessage;
        getMessageText = function () {
            var $message_input;
            $message_input = $('.message_input');
            return $message_input.val();
        };

        sendMessage = function (text) {
            var $messages, message;
            if (text.trim() === '') {
                return;
            }
            $('.message_input').val('');
            $messages = $('.messages');

            // Set message_side based on whether the message is from the user or chatbot
            var userMessageSide = 'user';
            var botMessageSide = 'bot';

            // Draw user message with user-message class
            message = new Message({
                text: text,
                message_side: userMessageSide
            });
            message.draw();

            // Call getResponse() to get the chatbot's response
            $.get("/get", { msg: text }).done(function(data) {
                // Draw bot message with bot-message class
                var botMessage = new Message({
                    text: data,
                    message_side: botMessageSide
                });
                botMessage.draw();
                $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
            });

            return $messages.animate({ scrollTop: $messages.prop('scrollHeight') }, 300);
        };

        $('.send_message').click(function (e) {
            return sendMessage(getMessageText());
        });
        $('.message_input').keyup(function (e) {
            if (e.which === 13) {
                return sendMessage(getMessageText());
            }
        });

        // Add initial bot message
        var initialBotMessage = new Message({
            text: 'How can I help you?',
            message_side: 'bot'
        });
        initialBotMessage.draw();
    });
})();
