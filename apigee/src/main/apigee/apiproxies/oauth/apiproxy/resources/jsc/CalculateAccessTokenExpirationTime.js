var id_token_expiration_timestamp = context.getVariable('jwt.VerifyJWT-FirebaseIDToken.decoded.claim.exp') * 1000;
var current_timestamp = context.getVariable('client.received.end.timestamp');
context.setVariable("access_token_expiration_time_ms", (id_token_expiration_timestamp - current_timestamp).toString());

